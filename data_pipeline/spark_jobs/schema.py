from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
)

# Synthetic "EHR events" schema (structured)
EHR_EVENTS_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("encounter_id", StringType(), False),
    StructField("event_time", TimestampType(), False),

    # basic demographics
    StructField("age", IntegerType(), True),
    StructField("sex", StringType(), True),

    # vitals
    StructField("heart_rate", DoubleType(), True),
    StructField("systolic_bp", DoubleType(), True),
    StructField("diastolic_bp", DoubleType(), True),
    StructField("spo2", DoubleType(), True),
    StructField("temperature_c", DoubleType(), True),
])

# Synthetic "clinical notes" schema (unstructured)
NOTES_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("encounter_id", StringType(), False),
    StructField("note_time", TimestampType(), False),
    StructField("clinical_note", StringType(), True),
])

# Labels schema (supervised learning target)
LABELS_SCHEMA = StructType([
    StructField("patient_id", StringType(), False),
    StructField("encounter_id", StringType(), False),
    StructField("label_readmit_30d", IntegerType(), True),  # 0/1
    StructField("label_mortality", IntegerType(), True),    # 0/1
])