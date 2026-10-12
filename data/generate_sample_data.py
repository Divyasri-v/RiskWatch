"""
Backup option: creates a synthetic dataset with the SAME columns as the
CDC Diabetes Health Indicators dataset, so the project runs even before
you download the real file from Kaggle.

Run:  python data/generate_sample_data.py           (100,000 rows)
      python data/generate_sample_data.py 250000    (custom size)

Use the real Kaggle dataset for your final results.
"""
import sys

import numpy as np
import pandas as pd

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
rng = np.random.default_rng(42)

age = rng.integers(1, 14, n)
sex = rng.integers(0, 2, n)
income = rng.integers(1, 9, n)
education = rng.integers(1, 7, n)
bmi = np.clip(rng.normal(28 + 0.2 * age, 6, n), 12, 80).round()
smoker = rng.binomial(1, 0.44, n)
activity = rng.binomial(1, np.clip(0.85 - 0.02 * age, 0.3, 0.9), n)
alcohol = rng.binomial(1, 0.06, n)
fruits = rng.binomial(1, 0.63, n)
veggies = rng.binomial(1, 0.81, n)
high_bp = rng.binomial(1, np.clip(0.05 + 0.05 * age + 0.01 * (bmi - 25), 0.02, 0.9), n)
high_chol = rng.binomial(1, np.clip(0.1 + 0.04 * age, 0.05, 0.8), n)
genhlth = np.clip((1 + 0.15 * age + 0.05 * (bmi - 25) - activity + rng.normal(0, 1, n)).round(), 1, 5)
ment = np.clip(rng.poisson(3, n) * rng.binomial(1, 0.3, n), 0, 30)
phys = np.clip((rng.poisson(4, n) * (genhlth >= 4)), 0, 30)
diffwalk = rng.binomial(1, np.clip(0.03 * genhlth + 0.01 * age - 0.05, 0.01, 0.8), n)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


diabetes = rng.binomial(1, sigmoid(-6.5 + 0.9 * high_bp + 0.6 * high_chol + 0.08 * bmi
                                   + 0.15 * age + 0.5 * genhlth - 0.1 * income))
heart = rng.binomial(1, sigmoid(-8 + 0.8 * high_bp + 0.7 * high_chol + 0.3 * smoker
                                + 0.25 * age + 0.5 * sex + 0.5 * genhlth + 0.5 * diabetes))
stroke = rng.binomial(1, sigmoid(-8.5 + 1.0 * high_bp + 0.2 * age + 0.4 * genhlth
                                 + 0.4 * diffwalk + 0.4 * heart))

df = pd.DataFrame({
    "Diabetes_binary": diabetes, "HighBP": high_bp, "HighChol": high_chol,
    "CholCheck": rng.binomial(1, 0.96, n), "BMI": bmi, "Smoker": smoker,
    "Stroke": stroke, "HeartDiseaseorAttack": heart, "PhysActivity": activity,
    "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": alcohol,
    "AnyHealthcare": rng.binomial(1, 0.95, n), "NoDocbcCost": rng.binomial(1, 0.08, n),
    "GenHlth": genhlth, "MentHlth": ment, "PhysHlth": phys, "DiffWalk": diffwalk,
    "Sex": sex, "Age": age, "Education": education, "Income": income,
}).astype(float)

out = "data/diabetes_binary_health_indicators_BRFSS2015.csv"
df.to_csv(out, index=False)
print(f"Wrote {n:,} synthetic rows to {out}")
for col in ["Diabetes_binary", "HeartDiseaseorAttack", "Stroke"]:
    print(f"  {col:<22} {df[col].mean():.1%} positive")
