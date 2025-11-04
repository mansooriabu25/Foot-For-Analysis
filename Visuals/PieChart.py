import plotly.express as px
import streamlit as st
import pandas as pd
import datetime
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import re

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

def get_mongo_data():
    if not MONGO_URI or not DB_NAME:
        st.error("❌ MongoDB configuration missing. Check your .env file.")
        return []
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return list(db["visuals"].find({}))

def parse_age_range(age_str):
    if not age_str or not isinstance(age_str, str):
        return None
    nums = re.findall(r'\d+', age_str)
    if len(nums) == 2:
        return (int(nums[0]) + int(nums[1])) / 2
    elif len(nums) == 1:
        return int(nums[0])
    return None

def PieChart():
    st.header("🎂 Age Group Distribution Pie Chart")

    filter_option = st.selectbox(
        "Select time range:",
        ["All time", "Today", "Last 30 minutes", "Last 1 hour", "Last 3 hours"],
        key="piechart_filter"
    )

    data = get_mongo_data()
    if not data:
        st.info("No visual logs available yet.")
        return

    df = pd.DataFrame(data)

    if "timestamp" not in df.columns:
        st.warning("⚠️ 'timestamp' field not found.")
        return

    if "meta" not in df.columns:
        st.warning("⚠️ 'meta' field not found.")
        return

    # Convert timestamp field
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

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

    # Extract age and gender from meta dict
    df["age"] = df["meta"].apply(lambda x: x.get("age") if isinstance(x, dict) else None)
    df["gender"] = df["meta"].apply(lambda x: x.get("gender") if isinstance(x, dict) else None)

    # Parse and filter age
    df["age_numeric"] = df["age"].apply(parse_age_range)
    df = df[df["age_numeric"].notnull()]

    if df.empty:
        st.info("No valid age data to plot.")
        return

    bins = [10, 20, 30, 40, 50, 60, 100]
    labels = ["10–20", "20–30", "30–40", "40–50", "50–60", "60+"]

    df["age_group"] = pd.cut(df["age_numeric"], bins=bins, labels=labels, right=False)

    age_group_counts = (
        df["age_group"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"index": "Age Group", "age_group": "Count"})
    )
    age_group_counts.columns = ["Age Group", "Count"]

    if age_group_counts.empty:
        st.info("No data to display in the pie chart.")
        return

    fig = px.pie(
        age_group_counts,
        names="Age Group",
        values=age_group_counts["Count"].tolist(),
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}",
        pull=[0.03] * len(age_group_counts),
    )
    fig.update_layout(
        showlegend=True,
        legend_title="Age Group",
        margin=dict(l=40, r=40, t=60, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)
