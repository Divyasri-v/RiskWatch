"""
RiskWatch - shared settings and helpers used by every module.
"""
import os

# ---------- Storage ----------
# Default: HDFS running in Docker. For low-RAM laptops set
# STORAGE_BASE=file:///app/storage to use local disk instead.
STORAGE_BASE = os.getenv("STORAGE_BASE", "hdfs://namenode:9000/riskwatch")
RAW_PATH = f"{STORAGE_BASE}/raw/health_indicators.csv"
TRAIN_PATH = f"{STORAGE_BASE}/processed/train.parquet"
TEST_PATH = f"{STORAGE_BASE}/processed/test.parquet"
MODEL_PATH = f"{STORAGE_BASE}/models"

LOCAL_DATA = os.getenv("LOCAL_DATA", "data/diabetes_binary_health_indicators_BRFSS2015.csv")
STREAM_SAMPLE = "data/stream_sample.csv"      # held-out rows the producer replays
METRICS_FILE = "data/model_metrics.json"      # training results shown on the dashboard

# ---------- Kafka / MongoDB ----------
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")
KAFKA_TOPIC = "patients"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017")
MONGO_DB = "riskwatch"
MONGO_COLLECTION = "predictions"

# ---------- Data schema ----------
# The 3 diseases we predict (these are label columns in the dataset)
DISEASES = {
    "diabetes": "Diabetes_binary",
    "heart": "HeartDiseaseorAttack",
    "stroke": "Stroke",
}

# Input features shared by all 3 models (labels are excluded to avoid leakage)
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income",
]

# ---------- Risk levels ----------
LOW_LIMIT = 0.30
HIGH_LIMIT = 0.60


def risk_level(prob: float) -> str:
    if prob >= HIGH_LIMIT:
        return "High"
    if prob >= LOW_LIMIT:
        return "Medium"
    return "Low"


AGE_GROUPS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
    12: "75-79", 13: "80+",
}


def get_spark(app_name: str, kafka: bool = False):
    """Create a Spark session. kafka=True adds the Spark-Kafka connector."""
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
    )
    if kafka:
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3",
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
