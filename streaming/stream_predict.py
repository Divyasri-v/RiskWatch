"""
Module 5: Spark Structured Streaming job.
Reads patient records from Kafka, predicts risk for 3 diseases,
and writes the results to MongoDB every few seconds.

Run:  python streaming/stream_predict.py
"""
from datetime import datetime

from pymongo import MongoClient
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, StructField, StructType, DoubleType

import config as C
from streaming.scoring import load_models, score


def ensure_topic():
    """Create the Kafka topic if it does not exist yet (Spark cannot read a missing topic)."""
    import time
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError

    for attempt in range(10):
        try:
            admin = KafkaAdminClient(bootstrap_servers=C.KAFKA_BOOTSTRAP)
            break
        except NoBrokersAvailable:
            print(f"Kafka not ready, retrying ({attempt + 1}/10)...")
            time.sleep(3)
    else:
        raise SystemExit("Could not connect to Kafka. Is docker compose running?")
    try:
        admin.create_topics([NewTopic(C.KAFKA_TOPIC, num_partitions=1, replication_factor=1)])
        print(f"Created Kafka topic '{C.KAFKA_TOPIC}'")
    except TopicAlreadyExistsError:
        print(f"Kafka topic '{C.KAFKA_TOPIC}' already exists")
    finally:
        admin.close()
    time.sleep(2)  # give Kafka a moment to elect the partition leader


ensure_topic()
spark = C.get_spark("RiskWatch-Stream", kafka=True)

print("Loading the 3 trained models...")
models = load_models()

# Schema of each JSON message sent by the producer
schema = StructType(
    [StructField(c, DoubleType()) for c in C.FEATURES]
    + [StructField(f"actual_{n}", IntegerType()) for n in C.DISEASES]
    + [StructField("patient_id", StringType()), StructField("event_time", StringType())]
)

# 1. Read the live stream from Kafka
raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", C.KAFKA_BOOTSTRAP)
    .option("subscribe", C.KAFKA_TOPIC)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

# 2. Kafka gives bytes -> convert to columns using the schema
patients = (
    raw.select(F.from_json(F.col("value").cast("string"), schema).alias("p"))
    .select("p.*")
    .dropna(subset=C.FEATURES)
)

# 3. Apply all 3 models
scored = score(patients, models)

mongo = MongoClient(C.MONGO_URI)
collection = mongo[C.MONGO_DB][C.MONGO_COLLECTION]


# 4. Every micro-batch: save results to MongoDB and print a summary
def write_batch(batch_df, batch_id):
    rows = [r.asDict() for r in batch_df.collect()]
    if not rows:
        return
    now = datetime.now()
    for r in rows:
        r["processed_at"] = now
    collection.insert_many(rows)
    alerts = sum(1 for r in rows if r["alert"])
    print(f"[batch {batch_id}] scored {len(rows)} patient(s), {alerts} high-risk alert(s)")
    for r in rows:
        if r["alert"]:
            print(f"   ALERT {r['patient_id']}: diabetes={r['diabetes_level']}, "
                  f"heart={r['heart_level']}, stroke={r['stroke_level']}")


query = (
    scored.writeStream.foreachBatch(write_batch)
    .option("checkpointLocation", f"{C.STORAGE_BASE}/checkpoints/stream")
    .trigger(processingTime="2 seconds")
    .start()
)
print(f"Streaming started. Listening on Kafka topic '{C.KAFKA_TOPIC}'. Ctrl+C to stop.")
query.awaitTermination()