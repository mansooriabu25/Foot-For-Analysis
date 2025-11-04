import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

def get_mongo_data():
    """Fetch all records from visuals collection"""
    if not MONGO_URI or not DB_NAME:
        st.error("❌ MongoDB configuration missing. Check your .env file.")
        return []
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return list(db["visuals"].find({}))

def BarChart():
    st.header("🧍 Hourly Gender Distribution Bar Chart (9 AM - 11 PM)")

    filter_option = st.selectbox(
        "Select time range:",
        ["All time", "Today", "Last 30 minutes", "Last 1 hour", "Last 3 hours"],
        key="barchart_filter"
    )

    # --- Fetch data ---
    data = get_mongo_data()
    if not data:
        st.info("No visual logs available yet.")
        return

    df = pd.DataFrame(data)

    # --- Fix timestamp field ---
    if "datetime_str" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime_str"], errors="coerce")
    else:
        st.warning("⚠️ No datetime_str found in MongoDB.")
        return

    now = datetime.datetime.now()
    if filter_option == "Today":
        df = df[df["timestamp"].dt.date == now.date()]
    elif filter_option == "Last 30 minutes":
        df = df[df["timestamp"] >= now - pd.Timedelta(minutes=30)]
    elif filter_option == "Last 1 hour":
        df = df[df["timestamp"] >= now - pd.Timedelta(hours=1)]
    elif filter_option == "Last 3 hours":
        df = df[df["timestamp"] >= now - pd.Timedelta(hours=3)]

    if df.empty:
        st.info("No data for the selected time range.")
        return

    # --- Filter between 9 AM and 11 PM ---
    df = df[(df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour <= 23)]

    df["hour_bin"] = df["timestamp"].dt.floor("h")
    df["hour_label"] = df["hour_bin"].dt.strftime("%H:%M")

    # Extract gender properly
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown")
    else:
        # Sometimes gender is inside meta dict
        df["gender"] = df["meta"].apply(
            lambda m: m.get("gender") if isinstance(m, dict) else "Unknown"
        )


    hourly_gender_counts = (
        df.groupby(["hour_label", "gender"])
        .size()
        .reset_index(name="Count")
    )

    # --- Hover times ---
    hover_data = (
        df.groupby(["hour_label", "gender"])["timestamp"]
        .apply(lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()))
        .reset_index()
        .rename(columns={"timestamp": "Exact_Times"})
    )

    hourly_gender_counts = pd.merge(
        hourly_gender_counts, hover_data, on=["hour_label", "gender"], how="left"
    )

    # --- Plot ---
    fig = px.bar(
        hourly_gender_counts,
        x="hour_label",
        y="Count",
        color="gender",
        barmode="group",
        hover_data=["Exact_Times"],
        color_discrete_map={"Male": "#71B4F8", "Female": "#F897C8", "Other": "#AAAAAA"}
    )

    fig.update_layout(
        xaxis_title="Hour of the Day",
        yaxis_title="Number of Visitors",
        plot_bgcolor="black",
        hoverlabel=dict(bgcolor="white", font_color="black"),
        margin=dict(l=60, r=40, t=60, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)
