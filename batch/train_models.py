"""
Module 3: Train one Random Forest per disease with Spark MLlib,
evaluate on the test set, and save the models to HDFS.

Run:  python batch/train_models.py
"""
import json

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

import config as C

spark = C.get_spark("RiskWatch-Train")

train = spark.read.parquet(C.TRAIN_PATH)
test = spark.read.parquet(C.TEST_PATH)

# Combine the input columns into one "features" vector (shared by all models)
assembler = VectorAssembler(inputCols=C.FEATURES, outputCol="features")
train = assembler.transform(train).cache()
test = assembler.transform(test).cache()
print(f"Training rows: {train.count():,}  |  Test rows: {test.count():,}\n")

metrics = {}
for name, label in C.DISEASES.items():
    print(f"=== Training {name.upper()} model (label: {label}) ===")

    # Class weights: positive cases are rare, so give them more weight
    n = train.count()
    pos = train.filter(F.col(label) == 1).count()
    w_pos, w_neg = n / (2 * pos), n / (2 * (n - pos))
    weighted = train.withColumn(
        "weight", F.when(F.col(label) == 1, w_pos).otherwise(w_neg)
    )

    rf = RandomForestClassifier(
        labelCol=label,
        featuresCol="features",
        weightCol="weight",
        numTrees=50,
        maxDepth=8,
        seed=42,
        # unique output columns so 3 models can run on the same DataFrame
        predictionCol=f"{name}_pred",
        probabilityCol=f"{name}_prob",
        rawPredictionCol=f"{name}_raw",
    )
    model = rf.fit(weighted)

    # Evaluate on unseen test data
    pred = model.transform(test)
    auc = BinaryClassificationEvaluator(
        labelCol=label, rawPredictionCol=f"{name}_raw", metricName="areaUnderROC"
    ).evaluate(pred)

    def multi(metric):
        return MulticlassClassificationEvaluator(
            labelCol=label, predictionCol=f"{name}_pred",
            metricName=metric, metricLabel=1.0,
        ).evaluate(pred)

    result = {
        "auc": round(auc, 4),
        "accuracy": round(multi("accuracy"), 4),
        "precision": round(multi("precisionByLabel"), 4),
        "recall": round(multi("recallByLabel"), 4),
    }
    # Top 5 most important features
    imp = sorted(zip(C.FEATURES, model.featureImportances.toArray()),
                 key=lambda x: -x[1])[:5]
    result["top_features"] = [[f, round(float(v), 4)] for f, v in imp]
    metrics[name] = result

    print(f"  AUC={result['auc']}  Accuracy={result['accuracy']}  "
          f"Precision={result['precision']}  Recall={result['recall']}")
    print(f"  Top features: {', '.join(f for f, _ in result['top_features'])}")

    path = f"{C.MODEL_PATH}/{name}"
    model.write().overwrite().save(path)
    print(f"  Saved model -> {path}\n")

with open(C.METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"All 3 models trained. Metrics saved to {C.METRICS_FILE}")
spark.stop()
