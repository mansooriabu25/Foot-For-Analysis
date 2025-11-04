import os
import time
import pickle
import threading
from collections import OrderedDict, deque
from datetime import datetime, timedelta
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from Visuals.LineChart import LineChart
from Visuals.BarChart import BarChart
from Visuals.PieChart import PieChart
from DB.db import get_visual_logs, clear_visual_logs, persons_col, visuals_col, save_person
from DB.db import ensure_files_and_db, load_db, save_db, log_attendance, log_visual_appearance,delete_all
from Detection_Cam import FaceAttendanceSystem
from Cam import  camera_thread_fn


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



# -------------------------- Streamlit UI --------------------------
st.set_page_config(page_title="Face Recognition ", layout="wide")
st.title("Foot-For-Analysis  — MTCNN + FaceNet (OpenCV window)")

ensure_files_and_db()

if "engine" not in st.session_state:
    st.session_state.engine = None
if "cam_thread" not in st.session_state:
    st.session_state.cam_thread = None
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "running" not in st.session_state:
    st.session_state.running = False

tabs = st.tabs(["Home","Visuals Dashboard"])

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

    st.title("📊 Visuals - Visitor Time Pattern")
    LineChart()
    BarChart()
    PieChart()

    if st.button("📊 Clear Visual Logs"):
        clear_visual_logs()
        st.success("Visual logs cleared successfully.")
    if st.button("🗑️ Delete All"):
        delete_all()
        st.success("Deleted Successfully")
st.markdown("---")
st.caption("Pipeline: MTCNN (detection) → FaceNet/InceptionResnetV1 (512-d embeddings) → centroid tracking → 1-minute save → daily attendance log.")