import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
# --- MongoDB Setup ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

def get_mongo_data():
    if not MONGO_URI or not DB_NAME:
        st.error("❌ MongoDB configuration missing. Check your .env file.")
        return []
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return list(db["visuals"].find({}))

def LineChart():
    st.header("🕒 Visitors per Hour (9 AM - 11 PM) Line Chart")

    # --- Fetch data from MongoDB ---
    data = get_mongo_data()
    if not data:
        st.info("No data found in MongoDB.")
        return

    df = pd.DataFrame(data)

    # --- Detect and convert time field ---
    time_cols = [c for c in df.columns if c.lower() in ["timestamp", "time", "datetime", "datetime_str", "added", "date"]]
    if time_cols:
        df["timestamp"] = pd.to_datetime(df[time_cols[0]], errors="coerce")
    else:
        st.warning(f"No timestamp-like column found. Columns: {df.columns.tolist()}")
        return

    # --- Filter options ---
    filter_option = st.selectbox(
        "Select time range:",
        ["All time", "Today", "Last 30 minutes", "Last 1 hour", "Last 3 hours"],
        key="linechart_filter"
    )

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

    # --- For "All time", group by hour of day (across all dates) ---
    if filter_option == "All time":
        df["hour"] = df["timestamp"].dt.hour  # Extract hour (0-23)
        # Group by hour of day
        grouped = (
            df.groupby("hour")
            .agg({
                "identifier": lambda x: ", ".join(map(str, x)),
                "timestamp": lambda x: ", ".join(x.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()),  # Include date in hover
                "_id": "count"
            })
            .reset_index()
            .rename(columns={"_id": "Count"})
        )
        # Create full hour range (9-23) and merge to fill missing hours with 0
        full_hours = list(range(9, 24))
        grouped = pd.merge(
            pd.DataFrame({"hour": full_hours}),
            grouped,
            on="hour",
            how="left"
        ).fillna({"Count": 0, "identifier": "", "timestamp": ""})
        grouped["time_range"] = grouped["hour"].astype(str) + ":00"  
    else:
        # For date-specific filters, use the original logic (group by full hour_bin for today)
        df["hour_bin"] = df["timestamp"].dt.floor("h")
        grouped = (
            df.groupby("hour_bin")
            .agg({
                "identifier": lambda x: ", ".join(map(str, x)),
                "timestamp": lambda x: ", ".join(x.dt.strftime("%H:%M:%S").tolist()),
                "_id": "count"
            })
            .reset_index()
            .rename(columns={"_id": "Count"})
        )
        # Merge with today's full hour range
        start_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        end_time = now.replace(hour=23, minute=0, second=0, microsecond=0)
        full_hour_range = pd.date_range(start=start_time, end=end_time, freq="h")
        grouped = pd.merge(
            pd.DataFrame({"hour_bin": full_hour_range}),
            grouped,
            on="hour_bin",
            how="left"
        ).fillna({"Count": 0, "identifier": "", "timestamp": ""})
        grouped["time_range"] = grouped["hour_bin"].dt.strftime("%H:%M")

    # --- Create Line Chart ---
    fig = px.line(
        grouped,
        x="time_range",
        y="Count",
        markers=True,
        hover_data={
            "identifier": True,
            "timestamp": True,
            "Count": True,
        }
    )

    # --- Style the line ---
    fig.update_traces(line=dict(width=3, color="#FC3407"), marker=dict(size=8, color="white"))

    # --- Chart Layout ---
    fig.update_layout(
        xaxis_title="Time Range (Hour)",
        yaxis_title="Number of Visitors",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        xaxis=dict(showgrid=True, gridcolor="gray", dtick=1),
        yaxis=dict(showgrid=True, gridcolor="gray", dtick=1),
        margin=dict(l=70, r=40, t=80, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)