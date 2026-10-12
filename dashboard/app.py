"""
Module 6: Live dashboard (Streamlit) reading predictions from MongoDB.

Run:  streamlit run dashboard/app.py   ->  open http://localhost:8501
"""
import json
import os
import time

import altair as alt
import pandas as pd
import streamlit as st
from pymongo import DESCENDING, MongoClient

import config as C

st.set_page_config(page_title="RiskWatch", page_icon="🩺", layout="wide")


@st.cache_resource
def get_collection():
    return MongoClient(C.MONGO_URI)[C.MONGO_DB][C.MONGO_COLLECTION]


col = get_collection()
NAMES = {"diabetes": "Diabetes", "heart": "Heart Disease", "stroke": "Stroke"}
LEVELS = ["Low", "Medium", "High"]

# ---------- Sidebar ----------
st.sidebar.title("RiskWatch")
refresh = st.sidebar.slider("Refresh every (seconds)", 2, 30, 3)
window = st.sidebar.slider("Records to analyse", 100, 5000, 1000, step=100)
live = st.sidebar.toggle("Live updates", value=True)
if st.sidebar.button("Clear all predictions"):
    col.delete_many({})
    st.sidebar.success("Cleared")

# ---------- Load data ----------
docs = list(col.find({}, {"_id": 0}).sort("processed_at", DESCENDING).limit(window))
total = col.estimated_document_count()

st.title("🩺 RiskWatch: Real-Time Multi-Disease Risk Prediction")
st.caption("Kafka → Spark Structured Streaming → Spark MLlib → MongoDB → Streamlit")

if not docs:
    st.info("Waiting for data... Start the producer and the streaming job.")
else:
    df = pd.DataFrame(docs)
    df["age_group"] = df["Age"].astype(int).map(C.AGE_GROUPS)

    # ---------- KPIs ----------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients scored (total)", f"{total:,}")
    k2.metric("High-risk alerts (window)", int(df["alert"].sum()))
    k3.metric("Alert rate", f"{df['alert'].mean():.1%}")
    k4.metric("Last update", pd.to_datetime(df["processed_at"].max()).strftime("%H:%M:%S"))

    # ---------- Risk distribution ----------
    st.subheader("Risk level distribution")
    dist = pd.DataFrame([
        {"Disease": NAMES[n], "Level": lvl, "Patients": int((df[f"{n}_level"] == lvl).sum())}
        for n in C.DISEASES for lvl in LEVELS
    ])
    chart = alt.Chart(dist).mark_bar().encode(
        x=alt.X("Disease:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Level:N", sort=LEVELS),
        y=alt.Y("Patients:Q"),
        color=alt.Color("Level:N", sort=LEVELS,
                        scale=alt.Scale(domain=LEVELS, range=["#2e9d5b", "#e0a526", "#d64545"])),
        tooltip=["Disease", "Level", "Patients"],
    ).properties(height=320)
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(chart, use_container_width=True)
    avg = pd.DataFrame({"Average risk": [df[f"{n}_risk"].mean() for n in C.DISEASES]},
                       index=[NAMES[n] for n in C.DISEASES])
    c2.dataframe(avg.style.format("{:.1%}"), use_container_width=True)
    c2.caption("Low < 30%  ·  Medium 30-60%  ·  High ≥ 60%")

    # ---------- Alerts over time ----------
    st.subheader("Alerts over time")
    ts = df.copy()
    ts["time"] = pd.to_datetime(ts["processed_at"]).dt.floor("10s")
    trend = ts.groupby("time").agg(patients=("patient_id", "count"),
                                   alerts=("alert", "sum"))
    st.line_chart(trend)

    # ---------- Alert panel ----------
    st.subheader("🚨 Latest high-risk patients")
    alerts = df[df["alert"]].head(15)
    show = ["patient_id", "event_time", "age_group", "BMI", "HighBP", "HighChol"] + \
           [f"{n}_risk" for n in C.DISEASES]
    if alerts.empty:
        st.write("No high-risk patients in this window.")
    else:
        st.dataframe(
            alerts[show].style.format({f"{n}_risk": "{:.0%}" for n in C.DISEASES})
            .background_gradient(subset=[f"{n}_risk" for n in C.DISEASES], cmap="Reds",
                                 vmin=0, vmax=1),
            use_container_width=True, hide_index=True,
        )

    # ---------- Live feed ----------
    st.subheader("Live patient feed")
    feed = ["patient_id", "event_time", "age_group"] + \
           [f"{n}_level" for n in C.DISEASES] + \
           [f"actual_{n}" for n in C.DISEASES]
    st.dataframe(df[feed].head(20), use_container_width=True, hide_index=True)
    st.caption("actual_* = true label from the dataset (1 = has the disease), shown for comparison")

# ---------- Model performance ----------
if os.path.exists(C.METRICS_FILE):
    with st.expander("Model performance (from batch training)"):
        m = json.load(open(C.METRICS_FILE))
        perf = pd.DataFrame({NAMES[n]: {k: v for k, v in m[n].items() if k != "top_features"}
                             for n in m}).T
        st.dataframe(perf, use_container_width=True)
        for n in m:
            st.write(f"**{NAMES[n]}** top features: " +
                     ", ".join(f for f, _ in m[n]["top_features"]))

if live:
    time.sleep(refresh)
    st.rerun()
