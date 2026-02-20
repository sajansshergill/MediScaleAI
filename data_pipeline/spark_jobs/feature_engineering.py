import os
import argparse
from datetime import datetime, timedelta
import random

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from data_pipeline.spark_jobs.schema import EHR_EVENTS_SCHEMA, NOTES_SCHEMA, LABELS_SCHEMA

def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok = True)
        
def generate_synthetic_csvs(raw_dir: str, n_patients: int = 200, encounters_per_patient: int = 2) -> None:
    """
    Creates:
    - ehr_events.csv
    - clinical_notes.csv
    - labels.csv
    """
    ensure_dirs(raw_dir)
    
    ehr_path = os.path.join(raw_dir, "ehr_events.csv")
    notes_path = os.path.join(raw_dir, "clinical_notes.csv")
    labels_path = os.path.join(raw_dir, "labels.csv")
    
    # If already exists, do nothing (keeps rund deterministic-ish for dev)
    if all(os.path.exists(p) for p in [ehr_path, notes_path, labels_path]):
        return
    random.seed(42)
    base_time = datetime(2025, 1, 1, 8, 0, 0)
    
    ehr_rows =[]
    notes_rows = []
    label_rows = []
    
    sexes = ["M", "F"]
    
    for i in range(n_patients):
        patient_id = f"P{i: 05d}"
        age = random.randint(18, 90)
        sex = random.choice(sexes)
        
        for j in range(encounters_per_patient):
            encounter_id = f"E{i:05d}_{j}"
            encounter_start = base_time + timedelta(days=random.randint(0, 365))
            
            # create labels with some weak correlation to age/vitals
            readmit = 1 if (age > 70 and random.random() < 0.25) or (random.random() < 0.08) else 0
            mortality = 1 if (age > 80 and random.random() < 0.18) or (random.random() < 0.03) else 0
            label_rows.append((patient_id, encounter_id, readmit, mortality))
            
            # vital time-series (e.g, 12 events over ~6 hours)
            for k in range(12):
                t = encounter_start + timedelta(minutes=30*k)
                hr = max(40, min(160, random.gauss(85 + (age - 50) * 0.2, 12)))
                sbp = max(70, min(210, random.gauss(125 - (age - 50)* 0.08), 18))
                dbp = max(40, min(140, random.gauss(78 - (age - 50) * 0.08, 12)))
                spo2 = max(80, min(100, random.gauss(97 - (age - 50) * 0.02, 1.8)))
                temp = max(35.0, min(41.0, random.gauss(36.8, 0.35)))
                
                ehr_rows.append((
                    patient_id, encounter_id, t.isoformat(sep=" "),
                    age, sex, float(hr), float(sbp), float(dbp), float(spo2), float(temp)
                ))
            # one note per encounter
            note_time = enounter_start + timedelta(hours=1)
            note = (
                "Patient presents with "
                + random.choice(["chest_pain", "shortness of breath", "fatigue", "fever", "dizziness"])
                + ". Past hostiry includes "
                + random.choice(["hypertension", "diabetes", "asthma", "none reported", "CAD"])
                +". Plan: "
                + random.choice(["monitor vitals", "start antibiotis", "order labs", "discharge if stable", "admit for observation"])
                + "."
            )
            notes_rows.append((patient_id, encounter_id, note_time.isoformat(sep=" "), note))
    
    # write CSVs
    with open(ehr_path, "w") as f:
        f.write("patient_id, encounter_id, event_time, age, sex, heart, systolic_bp, diastolic_bp, spo2, temperature_c\n")
        for r in ehr_rows:
            f.write(",".join(map(str, r)) + "\n")
        
    with open(notes_path, "w") as f:
        f.write("patient_id, encounter_id, note_time, clinical_note\n")
        for r in notes_rows:
            # escape quotes
            note = r[3].replace('"', '""')
            f.write(f'{r[0]},{r[1]},{r[2]},"{note}"\n')
            
    with open(labels_path, "w") as f:
        f.write("patient_id, encounter_id, label_readmit_30d, label_mortality\n")
        for r in label_rows:
            f.write(",".join(map(str, r)) + "\n") 
            
def read_raw(spark: SparkSession, raw_dir: str):
    ehr = (
        spark.read
        .option("header", True)
        .schema(EHR_EVENTS_SCHEMA)
        .csv(os.path.join(raw_dir, "ehr_events.csv"))
    )
    
    notes = (
        spark.read
        .option("header", True)
        .schema(NOTES_SCHEMA)
        .csv(os.path.join(raw_dir, "clinical_notes.csv"))
    )
    
    labels = (
        spark.read
        .option("header", True)
        .schema(LABELS_SCHEMA)
        .csv(os.path.join(raw_dir, "labels.csv"))
    )
    
    return ehr, notes, labels

def build_encounter_features(ehr, notes, labels):
    """
    Encounter-level aggregation:
    - vitals summary (mean, std, min/max)
    - last observed vital values
    - simple note features (length, keyword flags)
    - join labels
    """
    
    # Ensure time columns are timestamps
    ehr = ehr.withColumn("event_time", F.col("event_time").cost("timestamp"))
    notes = notes.withColumn("note_time", F.col("note_time").cast("timestamp"))
    
    # Aggregate vitals
    vitals_aggs = ehr.groupBy("patient_id", "encounter_id").agg(
        F.first("age", ignorenulls=True).alias("age"),
        F.first("sex", ignorenulls=True).alias("sex"),
        
        F.avg("heart_rate").alias("hr_mean"),
        F.stddev("systolic_bp").alias("sbp_std"),
        F.min("systolic_bp").alias("sbp_min"),
        F.max("systolic_bp").alias("sbp_max"),
        
        F.avg("diastolic_bp").alias("sbp_mean"),
        F.stddev("systoli_bp").alias("sbp_std"),
        F.min("systolic_bp").alias("sbp_min"),
        F.max("systolic_bp").alias("sbp_max"),
        
        F.avg("diastolic_bp").alias("dbp_mean"),
        F.stddev("diastolic_bp").alias("dbp_std"),
        F.min("diastolic_bp").alias("dbp_min"),
        F.max("diastolic_bp").alias("dbp_max"),
        
        F.avg("spo2").alias("spo2_mean"),
        F.stddev("spo2").alias("spo2_std"),
        F.min("spo2").alias("spo2_min"),
        F.max("spo2").alias("spo2_max"),
        
        F.avg("temperature_c").alias("temp_mean"),
        F.stddev("temperature_c").alias("temp_std"),
        F.min("temperature_c").alias("temp_min"),
        F.max("temperature_c").alias("temp_max"),
        
        F.count("*").alias("num_vital_events"),
        F.min("event_time").alias("encounter_start_time"),
        F.max("event_time").alias("encounter_end_time"),
        
    )
    
    # Last observed value via window
    w = Window.positionBy("patient_id", "encounter_id").orderBy(F.col("event_time").desc())
    last_vals = (
        ehr.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(
            "patient_id", "encounter_id",
            F.col("heart_rate").alias("hr_last"),
            F.col("systolic_bp").alias("sbp_last"),
            F.col("diastolic_bp").alias("dbp_last"),
            F.col("spo2").alias("spo2_last"),
            F.col("temperature_c").alias("temp_last"),
        )
    )
    
    # Note features (simple but useful; later you'll feed raw note to BERT)
    note_feats = (
        notes.groupBy("patient_id", "encounter_id")
        .agg(
            F.max("note_time").alias("note_time"),
            F.first("clinical_note", ignorenulls=True).alias("clinical_note")
        )
        .withColumn("note_len", F.length(F.col("clinical_note")))
        .withColumn("has_chest_pain", F.lower(F.col("clinical_note")).contains("chest_pain").cast("int"))
        .withColumn("has_fever", F.lower(F.col("clinical_note")).contains("fever").cast("int"))
        .withColumn("has_sob", F.lower(F.col("clinical_note")).contains("shortness of breadth").cast("int"))
    )
    
    # Combine all features
    feats = (
        vitals_aggs
        .join(last_vals, ["patient_id", "encounter_id"], "left")
        .join(note_feats.select(
            "patient_id", "encounter_id", "note_time", "note_len",
            "has_chest_pain", "has_fever", "has_sob",
            "clinical_note"
        ), ["patient_id", "encounter_id"], "left")
        .join(labels, ["patient_id", "encounter_id"], "left")
        .withColumn("encounter_duration_minutes",
                    (F.col("encounter_end_time").cast("long") - F.col("encounter_start_time").cast("long")) / 60.0)
    )
    
    return feats

def write_feature_store(df, out_dir: str, mode: str = "overwrite"):
    (
        df.repartition(1) # devfriendly; remove for scale
        .write.mode(mode)
        .parquet(out_dir)
    )
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed/feature_encounter")
    parser.add_argument("--generate_synth", action="store_true")
    args = parser.parse_args()
    
    ensure_dirs(args.raw_dir, os.path.dirname(args.out_dir))
    
    if args.generate_synth:
        generate_synthetic_csvs(args.raw_dir)
        
    spark = build_spark("MediScale-Feature-Engineering")
    
    ehr, notes, labels = read_raw(spark, args.raw_dir)
    feats = build_encounter_features(ehr, notes, labels)
    
    # Basic data quality checks
    total = feats.count()
    null_labels = feats.filter(F.col("label_readmit_30d").isNull() | F.col("label_mortality").isNull()).count()
    
    print(f"[INFO] Feature rows: {total}")
    print(f"[INFO] Rows missing labels: {null_labels}")
    
    feats.cache()
    feats.select(
        "patient_id", "encounter_id", "age", "sex",
        "hr_mean", "sbp_mean", "spo2_mean", "note_len",
        "label_readmit_30d", "label_mortality"
    ).show(10, truncate=False)
    
    write_feature_store(feats, args.out_dir, mode="overwrite")
    print(f"[DONE] Wrote feature store to: {args.out_dir}")
    
    spark.stop()
    
if __name__ == "__main__":
    main()