# RiskWatch: Real-Time Multi-Disease Risk Prediction System

Big Data Architecture final project. RiskWatch streams patient health records through Kafka, scores every patient for **Diabetes, Heart Disease and Stroke** risk using Spark MLlib models, stores the results in MongoDB, and shows them on a live Streamlit dashboard.

```
                 BATCH LAYER (run once)
 CSV dataset ──► HDFS ──► Spark (clean) ──► Spark MLlib ──► 3 models saved in HDFS
                                                               │
                 SPEED LAYER (real time)                       ▼
 producer.py ──► Kafka topic "patients" ──► Spark Structured Streaming
                                           (apply 3 models, assign risk level)
                                                               │
                 SERVING LAYER                                 ▼
                       Streamlit dashboard ◄── MongoDB (predictions)
```

## Tech stack

| Layer | Tool |
|---|---|
| Storage | HDFS (Hadoop 3.2) |
| Batch processing | Apache Spark 3.5 (PySpark) |
| Machine learning | Spark MLlib (Random Forest) |
| Ingestion | Apache Kafka 3.8 |
| Stream processing | Spark Structured Streaming |
| Database | MongoDB 7 |
| Dashboard | Streamlit |
| Deployment | Docker Compose |

Everything is free, open source, and runs on your own laptop.

## Project structure

```
riskwatch/
├── docker-compose.yml        # HDFS, Kafka, MongoDB and the app container
├── Dockerfile                # Python 3.11 + Java + PySpark
├── requirements.txt
├── config.py                 # all settings in one place
├── data/
│   └── generate_sample_data.py   # backup: synthetic data with the same columns
├── batch/
│   ├── preprocess.py         # ingest to HDFS, clean, split
│   └── train_models.py       # train + evaluate 3 models
├── streaming/
│   ├── producer.py           # sends patients to Kafka
│   ├── scoring.py            # shared prediction logic
│   └── stream_predict.py     # Spark Streaming -> MongoDB
├── dashboard/
│   └── app.py                # live dashboard
└── docs/
    └── RiskWatch_Explanation.pdf
```

---

## Requirements

- **Docker Desktop** (Windows, Mac or Linux)
- **8 GB RAM** minimum, 16 GB recommended
- **Fast internet for the first build only** (about 400 MB download). A phone hotspot works well if your Wi-Fi is slow.

You do **not** need to install Python, Java, Spark, Hadoop or Kafka yourself. Docker handles all of it.

---

## Part A: One-time setup

### Step 1: Install Docker Desktop
1. Download it from **docker.com/products/docker-desktop** and install with default options. Keep **"Use WSL 2"** ticked on Windows.
2. Restart the laptop.
3. Open Docker Desktop and wait until it shows **"Engine running"**.

### Step 2: Set Docker memory (Windows, only for 8 GB RAM laptops)
Docker on Windows uses WSL 2, so memory is set in a file, not in Docker Desktop.

Press **Win + R**, type `notepad %UserProfile%\.wslconfig`, press Enter, paste this and save:
```ini
[wsl2]
memory=5GB
processors=4
swap=4GB
```
Then run `wsl --shutdown` in PowerShell and reopen Docker Desktop. Skip this step with 16 GB RAM.

### Step 3: Add the dataset
1. On Kaggle, search **"Diabetes Health Indicators Dataset"** (by Alex Teboul) and download it.
2. Unzip it. It contains 3 CSV files. Use only **`diabetes_binary_health_indicators_BRFSS2015.csv`** (the large one, about 22 MB, 253,680 rows). Do not use the "5050split" file.
3. Copy it into the `data/` folder.

No Kaggle account yet? You can test with synthetic data that has the same columns (after Step 4):
```bash
docker compose exec app python data/generate_sample_data.py
```
Use the real dataset for your final results.

### Step 4: Build and start all services
Open PowerShell (or the VS Code terminal) inside the `riskwatch` folder:
```bash
docker compose up -d --build
```
The first build downloads PySpark (317 MB). The screen may look stuck on `Downloading pyspark...` because there is no progress bar. Check Task Manager → Performance → Wi-Fi to confirm it is downloading.

Check that everything is up:
```bash
docker compose ps
```
All 5 containers should be running: `namenode`, `datanode`, `kafka`, `mongo`, `riskwatch-app`.

Open **http://localhost:9870** to see the HDFS web UI. It should show **Live Nodes: 1**.

**Wait about 1 minute** after starting so HDFS can finish booting.

### Step 5: Preprocess the data (batch layer)
```bash
docker compose exec app python batch/preprocess.py
```
Uploads the CSV into HDFS, removes nulls and duplicates, shows disease prevalence, splits 80/20 into train/test, and saves them as Parquet in HDFS.

Check in the HDFS UI: **Utilities → Browse the file system → /riskwatch**.

### Step 6: Train the 3 models
```bash
docker compose exec app python batch/train_models.py
```
Trains a Random Forest for each disease (3 to 8 minutes), prints AUC, Accuracy, Precision, Recall and top features, and saves the models to HDFS. Take a screenshot of this output for your report.

Part A is done. Data and models stay saved after shutdown.

---

## Part B: Run the live system

Open **3 terminals** inside the `riskwatch` folder. In VS Code, use the split terminal button.

### Step 7: Terminal 1, start the streaming job
```bash
docker compose exec app python streaming/stream_predict.py
```
It creates the Kafka topic automatically, loads the 3 models, and downloads the Spark-Kafka connector on the first run (about 1 minute). Wait until you see:
```
Streaming started. Listening on Kafka topic 'patients'. Ctrl+C to stop.
```

### Step 8: Terminal 2, send patients
```bash
docker compose exec app python streaming/producer.py --rate 2
```
`--rate 2` sends 2 patients per second. Terminal 1 starts printing batches and **ALERT** lines for high-risk patients:
```
[batch 76] scored 4 patient(s), 1 high-risk alert(s)
   ALERT P-88578: diabetes=High, heart=High, stroke=High
```

### Step 9: Terminal 3, start the dashboard
```bash
docker compose exec app streamlit run dashboard/app.py --server.address 0.0.0.0
```
Then **open http://localhost:8501 in your browser yourself**. It will not open automatically because Streamlit runs inside Docker, and the `http://0.0.0.0:8501` link it prints does not work in a Windows browser.

The dashboard refreshes every few seconds and shows:
- total patients scored and number of alerts
- risk level distribution per disease
- alerts over time
- latest high-risk patients
- live patient feed and model performance

While a Spark job runs, the Spark UI is at **http://localhost:4040**.

### Step 10: Stop everything
Press `Ctrl + C` in each terminal, then:
```bash
docker compose down        # stop (keeps data and models)
docker compose down -v     # stop and delete ALL stored data (you must redo Steps 5 and 6)
```

---

## Next time you run it
Skip Part A. Just run:
```bash
docker compose up -d
```
Wait 1 minute, then do Steps 7, 8 and 9.

---

## Normal messages (not errors)

| Message | Meaning |
|---|---|
| Long Python/Java error after pressing `Ctrl + C` | Spark always prints this when stopped |
| `WARN DAGScheduler: Broadcasting large task binary` | The models are a few MB, nothing wrong |
| `WARN ... falling behind` | Your laptop is busy, it catches up |
| `project has been loaded without an explicit name` | Harmless Docker Compose warning |
| VS Code: `Import "pyspark" could not be resolved` | PySpark is installed inside Docker, not on Windows. The code still runs |

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails with `Read timed out` | Slow internet. Run `docker compose up -d --build` again (downloads resume), or use a phone hotspot |
| `Name node is in safe mode` | Wait 1 minute and retry, or run `docker exec namenode hdfs dfsadmin -safemode leave` |
| `Could not connect to Kafka` | Wait 30 seconds after `docker compose up`, then retry |
| `Path does not exist ... .csv` | The CSV must be in `data/` with the exact name above |
| `Path does not exist ... models` | Run Steps 5 and 6 before Step 7 |
| Dashboard says "Waiting for data" | Make sure Terminal 1 (stream) and Terminal 2 (producer) are both running |
| Browser shows "site can't be reached" | Use `http://localhost:8501`, not `0.0.0.0` |
| Datanode keeps restarting | Run `docker compose down -v`, then `docker compose up -d`, then redo Steps 5 and 6 |
| Port 8501 / 9092 already in use | Close the other app using it, or change the port in `docker-compose.yml` |
| Very slow / out of memory | Close Chrome tabs, increase WSL memory (Step 2), or use the Low-RAM option below |

## Low-RAM option (8 GB laptops)
If HDFS makes your laptop too slow, store files on local disk instead. Everything else stays the same:
```bash
docker compose stop namenode datanode
docker compose exec -e STORAGE_BASE=file:///app/storage app python batch/preprocess.py
docker compose exec -e STORAGE_BASE=file:///app/storage app python batch/train_models.py
docker compose exec -e STORAGE_BASE=file:///app/storage app python streaming/stream_predict.py
```
Producer and dashboard commands do not change. In your report, mention HDFS as the production storage layer.

---

## Risk levels
| Probability | Level |
|---|---|
| below 30% | Low |
| 30% to 60% | Medium |
| 60% and above | High (alert) |

## Note
The dataset is from a public health survey (CDC BRFSS 2015), not clinical tests. RiskWatch is a screening and prioritisation tool, not a diagnosis tool.