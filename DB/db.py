import os
import numpy as np
import pandas as pd
from pymongo import MongoClient, ASCENDING
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId
from dotenv import load_dotenv

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "face_attendance")

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()  # force test connection
    db = client[DB_NAME]
    print("✅ MongoDB connection established.")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    db = None

# Collections
persons_col = db["persons"]
attendance_col = db["records"]
visuals_col = db["visuals"]

# Indexes for performance
persons_col.create_index([("person_id", ASCENDING)], unique=True)
attendance_col.create_index([("person_id", ASCENDING), ("date", ASCENDING)])
visuals_col.create_index([("timestamp", ASCENDING)])
visuals_col.create_index([("identifier", ASCENDING)])


# ------------------------------
# Embeddings database (replaces embeddings.pkl)
# ------------------------------
def ensure_files_and_db():
    # Just ensure collections exist by referencing them
    _ = db["records"]
    _ = db["visuals_log"]


def load_db():
    """Return embeddings, ids, names, and meta as dict (same structure as pickle)"""
    coll = db["embeddings"]
    records = list(coll.find({}, {"_id": 0}))
    if not records:
        return {"ids": [], "embeddings": [], "names": [], "meta": []}

    ids = [r["person_id"] for r in records]
    embeddings = [np.array(r["embedding"]) for r in records]
    names = [r.get("name", "") for r in records]
    meta = [r.get("meta", {}) for r in records]
    return {"ids": ids, "embeddings": embeddings, "names": names, "meta": meta}


def save_db(db_dict):
    """Upserts all records back into Mongo"""
    coll = db["embeddings"]
    coll.delete_many({})
    for pid, emb, name, meta in zip(
        db_dict["ids"], db_dict["embeddings"], db_dict["names"], db_dict["meta"]
    ):
        coll.insert_one({
            "person_id": pid,
            "embedding": emb.tolist(),
            "name": name,
            "meta": meta,
        })


# ------------------------------
# Attendance and Visual logs
# ------------------------------
def log_attendance(person_id):
    coll = db["records"]
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    already = coll.find_one({"person_id": person_id, "date": date_str})
    if not already:
        coll.insert_one({
            "person_id": person_id,
            "date": date_str,
            "time": time_str
        })


def log_visual_appearance(identifier):
    coll = db["visuals_log"]
    already = coll.find_one({"identifier": identifier})
    if not already:
        now = datetime.now()
        coll.insert_one({
            "identifier": identifier,
            "timestamp": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d")
        })


# -------------------------------------------------------------------
# 🧍 Persons (Face Data)
# -------------------------------------------------------------------
def load_all_persons():
    """Return all persons from MongoDB like old pickle structure."""
    docs = list(persons_col.find({}))
    if not docs:
        return {"ids": [], "embeddings": [], "meta": []}

    ids = [d["person_id"] for d in docs]
    embeddings = [np.array(d["embedding"], dtype=np.float32) for d in docs]
    meta = [d.get("meta", {}) for d in docs]
    return {"ids": ids, "embeddings": embeddings, "meta": meta}


def save_person(person_id, age=None, gender=None, embedding=None):
    persons_col.insert_one({
        "person_id": person_id,
        "age": age,
        "gender": gender,
        "embedding": embedding.tolist() if embedding is not None else None,
        "datetime_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def match_embedding(embedding, threshold=0.65):
    """Compare a new embedding to saved embeddings and return the best match."""
    docs = list(persons_col.find({}, {"person_id": 1, "embedding": 1}))
    if not docs:
        return None, 0.0

    saved = np.vstack([np.array(d["embedding"], dtype=np.float32) for d in docs])
    sims = cosine_similarity(embedding.reshape(1, -1), saved)[0]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return docs[best_idx]["person_id"], best_sim
    return None, best_sim


# def delete_person_everywhere(person_id):
#     """Delete a person from all collections."""
#     persons_col.delete_one({"person_id": person_id})
#     attendance_col.delete_many({"person_id": person_id})
#     visuals_col.delete_many({"identifier": person_id})


# -------------------------------------------------------------------
# 👁️ Visual Logs
# -------------------------------------------------------------------
def log_visual_appearance(identifier, meta=None):
    """Insert visual detection log into MongoDB with safe defaults."""
    ts = datetime.now()
    if meta is None:
        meta = {}

    # Try to fetch person info from persons_col
    person = persons_col.find_one({"person_id": identifier})
    if person:
        meta_data = person.get("meta", {})
        # Merge meta priorities: passed meta > stored meta
        meta = {**meta_data, **meta}

    # Default fallbacks if still missing
    meta.setdefault("age", "Unknown")
    meta.setdefault("gender", "Unknown")

    visuals_col.insert_one({
        "identifier": identifier,
        "datetime_str": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": ts,
        "meta": meta
    })


def get_visual_logs(time_filter=None):
    """Fetch visual logs with optional time filter."""
    now = datetime.now()
    query = {}

    if time_filter == "today":
        query["timestamp"] = {"$gte": datetime(now.year, now.month, now.day)}
    elif time_filter == "30m":
        query["timestamp"] = {"$gte": now - timedelta(minutes=30)}
    elif time_filter == "1h":
        query["timestamp"] = {"$gte": now - timedelta(hours=1)}
    elif time_filter == "3h":
        query["timestamp"] = {"$gte": now - timedelta(hours=3)}

    return list(visuals_col.find(query).sort("timestamp", 1))


def clear_visual_logs():
    """Delete all visual logs."""
    visuals_col.delete_many({})
def delete_all():
    persons_col.delete_many({})
    attendance_col.delete_many({})

def get_visual_dataframe():
    """Return visuals collection as pandas DataFrame."""
    docs = list(visuals_col.find({}))
    if not docs:
        return pd.DataFrame(columns=["identifier", "timestamp", "gender", "age", "datetime_str"])

    # Flatten meta fields (like gender, age)
    for d in docs:
        meta = d.get("meta", {})
        d["gender"] = meta.get("gender", "")
        d["age"] = meta.get("age", None)
        d.pop("_id", None)

    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
