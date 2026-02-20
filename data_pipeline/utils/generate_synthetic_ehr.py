from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from faker import Faker


fake = Faker()


@dataclass
class GenConfig:
    n_patients: int = 2000
    visits_per_patient_min: int = 1
    visits_per_patient_max: int = 4
    seed: int = 42
    start_date: str = "2024-01-01"
    end_date: str = "2025-12-31"


CHIEF_COMPLAINTS = [
    "chest pain", "shortness of breath", "fever", "abdominal pain",
    "headache", "dizziness", "weakness", "cough", "nausea", "back pain"
]

COMORBIDITIES = [
    "diabetes", "hypertension", "copd", "ckd", "asthma", "cad", "afib", "obesity"
]

MEDS = ["metformin", "lisinopril", "atorvastatin", "albuterol", "insulin", "aspirin", "warfarin"]


def _rand_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, max(delta.days, 1)))


def _clinical_note(age: int, complaint: str, comorbs: List[str], hr: int, sbp: int, dbp: int, wbc: float) -> str:
    comorb_txt = ", ".join(comorbs) if comorbs else "none"
    meds = ", ".join(random.sample(MEDS, k=random.randint(0, 3))) if random.random() < 0.7 else "none"
    return (
        f"HPI: {age} y/o presents with {complaint}. "
        f"PMH: {comorb_txt}. Medications: {meds}. "
        f"Vitals: HR {hr}, BP {sbp}/{dbp}. Labs: WBC {wbc:.1f}. "
        f"Assessment: monitor and treat per protocol."
    )


def generate(cfg: GenConfig) -> pd.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    start = datetime.fromisoformat(cfg.start_date)
    end = datetime.fromisoformat(cfg.end_date)

    rows = []
    for pid in range(1, cfg.n_patients + 1):
        base_age = int(np.clip(np.random.normal(58, 16), 18, 95))
        sex = random.choice(["F", "M"])
        n_visits = random.randint(cfg.visits_per_patient_min, cfg.visits_per_patient_max)

        for v in range(n_visits):
            visit_id = f"V{pid:05d}_{v+1}"
            visit_dt = _rand_date(start, end)

            complaint = random.choice(CHIEF_COMPLAINTS)
            comorbs = random.sample(COMORBIDITIES, k=random.randint(0, 3))

            # Vitals/labs with light signal for risk
            hr = int(np.clip(np.random.normal(84, 14), 45, 160))
            sbp = int(np.clip(np.random.normal(128, 18), 70, 220))
            dbp = int(np.clip(np.random.normal(78, 12), 40, 140))
            wbc = float(np.clip(np.random.normal(8.5, 2.5), 2.0, 25.0))
            lactate = float(np.clip(np.random.normal(1.6, 0.8), 0.5, 8.0))

            # "Ground truth" risk labels (synthetic)
            high_risk_score = (
                0.02 * (base_age - 50)
                + 0.015 * (hr - 80)
                + 0.01 * (wbc - 8)
                + 0.25 * (lactate - 1.5)
                + (0.35 if "ckd" in comorbs else 0)
                + (0.25 if "cad" in comorbs else 0)
                + (0.20 if "copd" in comorbs else 0)
            )
            prob_readmit_30d = float(1 / (1 + np.exp(-high_risk_score)))
            readmit_30d = int(np.random.rand() < prob_readmit_30d)

            # Synthetic LOS and mortality
            los_days = int(np.clip(np.random.gamma(2.0, 1.2) + (2 if readmit_30d else 0), 1, 20))
            mortality_prob = float(0.01 + 0.02 * max(0, (base_age - 70) / 10) + 0.03 * max(0, (lactate - 2.0)))
            mortality_inpatient = int(np.random.rand() < mortality_prob)

            note = _clinical_note(base_age, complaint, comorbs, hr, sbp, dbp, wbc)

            rows.append(
                {
                    "patient_id": pid,
                    "visit_id": visit_id,
                    "visit_ts": visit_dt.isoformat(),
                    "age": base_age,
                    "sex": sex,
                    "chief_complaint": complaint,
                    "comorbidities": comorbs,
                    "hr": hr,
                    "sbp": sbp,
                    "dbp": dbp,
                    "wbc": wbc,
                    "lactate": lactate,
                    "los_days": los_days,
                    "readmit_30d": readmit_30d,
                    "mortality_inpatient": mortality_inpatient,
                    "clinical_note": note,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    cfg = GenConfig()
    df = generate(cfg)

    os.makedirs("data/raw", exist_ok=True)
    out = "data/raw/ehr_visits.csv"
    df.to_csv(out, index=False)
    print(f"✅ Wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()