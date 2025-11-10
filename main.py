import os
import threading
from collections import OrderedDict, deque
import pandas as pd
import streamlit as st
import torch
from pymongo import MongoClient
from dotenv import load_dotenv
from Visuals.LineChart import LineChart
from Visuals.BarChart import BarChart
from Visuals.PieChart import PieChart
from DB.db import ensure_files_and_db,delete_all,clear_visual_logs
from Detection import FaceAttendanceSystem
from Cam import  camera_thread_fn

# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Face Recognition ", layout="wide")
st.title("Footfall-Analysis  — MTCNN + FaceNet (OpenCV window)")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ensure_files_and_db()

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Function to get MongoDB data from a specific collection
def get_mongo_data(collection_name):
    if not MONGO_URI or not DB_NAME:
        st.error("❌ MongoDB configuration missing. Check your .env file.")
        return []
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        return list(db[collection_name].find({}))
    except Exception as e:
        st.error(f"❌ Error connecting to MongoDB: {e}")
        return []

if "engine" not in st.session_state:
    st.session_state.engine = None
if "cam_thread" not in st.session_state:
    st.session_state.cam_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "running" not in st.session_state:
    st.session_state.running = False

tabs = st.tabs(["Home","Persons","Records","Visuals Dashboard"])

with tabs[0]:
    st.header("Home — Camera control (OpenCV window)")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Start (OpenCV window)"):
            already_running = (
                st.session_state.running
                and st.session_state.cam_thread is not None
                and getattr(st.session_state.cam_thread, "is_alive", lambda: False)()
            )
            if already_running:
                st.info("Camera already running.")
            else:
                st.session_state.engine = FaceAttendanceSystem(device=DEVICE)
                st.session_state.stop_event = threading.Event()
                th = threading.Thread(target=camera_thread_fn, args=(st.session_state.engine, st.session_state.stop_event), daemon=True)
                st.session_state.cam_thread = th
                st.session_state.running = True
                th.start()
                st.success("Camera started (OpenCV window). Close the window or press Stop to end.")
    with col2:
        if st.button("⏹️ Stop"):
            if not st.session_state.running:
                st.info("Camera is not running.")
            else:
                if st.session_state.stop_event is not None:
                    st.session_state.stop_event.set()
                th = st.session_state.cam_thread
                if th is not None:
                    th.join(timeout=3)
                st.session_state.running = False
                st.success("Stopping camera... (close OpenCV window if still open)")

    st.markdown("""
    **Notes**
    - The live video appears in a separate OpenCV window named: **Face Attendance (OpenCV)**.
    - Press **q** in that window to stop it, or use the **Stop** button here.
    - Streamlit UI will remain responsive while the OpenCV window is open.
    """)

with tabs[1]:
    st.header("👤 Person Data")
    # Fetch data
    persons_data = get_mongo_data("persons")
    
    if persons_data:
        df_persons = pd.DataFrame(persons_data)
        # Drop MongoDB's _id if present (optional, for cleaner display)
        if "_id" in df_persons.columns:
            df_persons = df_persons.drop(columns=["_id"])
        st.dataframe(df_persons)  
    else:
        st.info("No data found in the 'persons' collection.")
    # st.markdown("---")
    # st.subheader("🗑️ Delete Person")
    # del_pid = st.selectbox("Choose person to delete", options=df_persons['person_id'], key="del_select")
    # if st.button("Delete selected person"):
    #     delete_all()
    #     st.success(f"Deleted {del_pid}")
    #     db = load_db()

with tabs[2]:
    st.header("📊 Records Data")
    # Fetch data
    records_data = get_mongo_data("records")
    
    if records_data:
        df_records = pd.DataFrame(records_data)
        # Drop MongoDB's _id if present (optional, for cleaner display)
        if "_id" in df_records.columns:
            df_records = df_records.drop(columns=["_id"])
        st.dataframe(df_records)  
    else:
        st.info("No data found in the 'records' collection.")

with tabs[3]:

    st.title("📊 Visuals - Visitor Time Pattern")
    LineChart()
    BarChart()
    PieChart()

    if st.button("📊 Clear Visual Logs"):
        clear_visual_logs()
        st.success("Visual logs cleared successfully.")
    if st.button("🗑️ Delete All Records"):
        delete_all()
        st.success("Deleted Successfully")

st.markdown("---")
st.caption("Pipeline: MTCNN (detection) → FaceNet/InceptionResnetV1 (512-d embeddings) → centroid tracking → 10 Sec save → daily attendance log.")