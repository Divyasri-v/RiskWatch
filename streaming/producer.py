"""
Module 4: Simulates hospital check-ins by sending patient records
to the Kafka topic "patients", one by one.

Run:  python streaming/producer.py            (1 patient per second)
      python streaming/producer.py --rate 5   (5 patients per second)
"""
import argparse
import json
import random
import time
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

import config as C

parser = argparse.ArgumentParser()
parser.add_argument("--rate", type=float, default=1.0, help="patients per second")
parser.add_argument("--count", type=int, default=0, help="stop after N (0 = forever)")
args = parser.parse_args()

df = pd.read_csv(C.STREAM_SAMPLE)
print(f"Loaded {len(df):,} held-out patient records from {C.STREAM_SAMPLE}")

# Kafka may take a few seconds to start, so retry
for attempt in range(10):
    try:
        producer = KafkaProducer(
            bootstrap_servers=C.KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        break
    except NoBrokersAvailable:
        print(f"Kafka not ready, retrying ({attempt + 1}/10)...")
        time.sleep(3)
else:
    raise SystemExit("Could not connect to Kafka. Is docker compose running?")

print(f"Sending to topic '{C.KAFKA_TOPIC}' at {args.rate} patient(s)/sec. Ctrl+C to stop.\n")
sent = 0
try:
    while args.count == 0 or sent < args.count:
        row = df.iloc[random.randrange(len(df))]
        record = {c: float(row[c]) for c in C.FEATURES}
        # true labels travel along only so the dashboard can compare
        for name, label in C.DISEASES.items():
            record[f"actual_{name}"] = int(row[label])
        record["patient_id"] = f"P-{random.randint(10000, 99999)}"
        record["event_time"] = datetime.now().isoformat(timespec="seconds")

        producer.send(C.KAFKA_TOPIC, record)
        sent += 1
        print(f"[{sent:>5}] {record['patient_id']}  age={C.AGE_GROUPS[int(record['Age'])]:<6} "
              f"BMI={record['BMI']:.0f}  HighBP={int(record['HighBP'])}")
        time.sleep(1 / args.rate)
except KeyboardInterrupt:
    pass
finally:
    producer.flush()
    print(f"\nStopped. Sent {sent} patient records.")
