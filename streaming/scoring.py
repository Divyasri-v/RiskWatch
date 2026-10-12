"""
Shared scoring logic: load the 3 trained models and apply them to a
DataFrame (works for both batch and streaming DataFrames).
"""
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config as C


def load_models():
    return {
        name: RandomForestClassificationModel.load(f"{C.MODEL_PATH}/{name}")
        for name in C.DISEASES
    }


def level_col(prob_col):
    return (F.when(prob_col >= C.HIGH_LIMIT, "High")
             .when(prob_col >= C.LOW_LIMIT, "Medium")
             .otherwise("Low"))


def score(df, models):
    """Add <disease>_risk (probability 0-1) and <disease>_level for each disease."""
    df = VectorAssembler(inputCols=C.FEATURES, outputCol="features").transform(df)
    for name, model in models.items():
        df = model.transform(df)
        prob = F.round(vector_to_array(F.col(f"{name}_prob"))[1], 4)
        df = df.withColumn(f"{name}_risk", prob)
        df = df.withColumn(f"{name}_level", level_col(F.col(f"{name}_risk")))

    levels = [F.col(f"{n}_level") for n in C.DISEASES]
    df = df.withColumn("alert", F.array_contains(F.array(*levels), "High"))
    df = df.withColumn("high_risk_count",
                       sum(F.when(l == "High", 1).otherwise(0) for l in levels))

    keep = (["patient_id", "event_time"] + C.FEATURES
            + [f"actual_{n}" for n in C.DISEASES]
            + [f"{n}_risk" for n in C.DISEASES]
            + [f"{n}_level" for n in C.DISEASES]
            + ["alert", "high_risk_count"])
    return df.select(*keep)
