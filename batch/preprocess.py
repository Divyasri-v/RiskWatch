"""
Module 1 + 2: Ingest the raw CSV into HDFS, clean it with Spark,
split into train/test, and save as Parquet.

Run:  python batch/preprocess.py
"""
import os
from pyspark.sql import functions as F
import config as C
spark = C.get_spark("RiskWatch-Preprocess")

# 1. Ingest: local CSV -> HDFS (raw zone)
local_csv = "file://" + os.path.abspath(C.LOCAL_DATA)
print(f"[1/5] Reading local file {C.LOCAL_DATA}")
raw = spark.read.csv(local_csv, header=True, inferSchema=True)
raw.write.mode("overwrite").option("header", True).csv(C.RAW_PATH)
print(f"      Stored raw data in {C.RAW_PATH}")

# 2. Read back from HDFS (this is what a real pipeline would do)
df = spark.read.csv(C.RAW_PATH, header=True, inferSchema=True)
total = df.count()
print(f"[2/5] Rows in raw zone: {total:,}  |  Columns: {len(df.columns)}")

# 3. Clean
all_cols = C.FEATURES + list(C.DISEASES.values())
df = df.select([F.col(c).cast("double").alias(c) for c in all_cols])
df = df.dropna().dropDuplicates()
clean = df.count()
print(f"[3/5] After removing nulls and duplicates: {clean:,} rows "
      f"({total - clean:,} removed)")

# 4. Quick look at class balance for each disease
print("[4/5] Disease prevalence (share of positive cases):")
for name, label in C.DISEASES.items():
    rate = df.agg(F.avg(label)).first()[0]
    print(f"      {name:<9} {rate:6.2%}")

# 5. Train/test split and save as Parquet (columnar, compressed)
train, test = df.randomSplit([0.8, 0.2], seed=42)
train.write.mode("overwrite").parquet(C.TRAIN_PATH)
test.write.mode("overwrite").parquet(C.TEST_PATH)

# Keep a small held-out sample locally for the real-time producer
test.limit(5000).toPandas().to_csv(C.STREAM_SAMPLE, index=False)

print(f"[5/5] Train: {train.count():,} rows -> {C.TRAIN_PATH}")
print(f"      Test : {test.count():,} rows -> {C.TEST_PATH}")
print(f"      Stream sample (5,000 rows) -> {C.STREAM_SAMPLE}")
spark.stop()
