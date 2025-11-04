import os
from collections import OrderedDict, deque
from datetime import datetime, timedelta
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from DB.db import get_visual_logs, clear_visual_logs, persons_col, visuals_col, save_person
from DB.db import ensure_files_and_db, load_db, save_db, log_attendance, log_visual_appearance,delete_all
from utils import Track, predict_age_gender,compute_cosine_sim


# -------------------------- Core Config --------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VIDEO_SRC = 0
DETECTION_EVERY_N_FRAMES = 10
MIN_SECONDS_TO_SAVE = 10
SIMILARITY_THRESHOLD = 0.7
TRACKER_MAX_DISAPPEAR = 10
OUTPUT_WINDOW_NAME = "Face Attendance (OpenCV) - press 'q' or Stop in UI"
# -------------------------- Visual --------------------------
BOX_COLOR = (0, 255, 0)
UNSAVED_COLOR = (0, 165, 255)
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -------------------------- Core face attendance system with smooth tracking --------------------------
class FaceAttendanceSystem:
    def __init__(self, device=DEVICE):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.tracks = OrderedDict()
        self.next_track_id = 0
        self.frame_index = 0
        self.prev_gray = None
        ensure_files_and_db()
        self.db = load_db()
        self.logged_visuals = set()

    def detect_faces(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        crops, boxes_out = [], []

        if boxes is None:
            return [], []

        h, w = frame_bgr.shape[:2]
        for box in boxes:
            if box is None or any(b is None for b in box):
                continue
            try:
                x1, y1, x2, y2 = [int(b) for b in box]
            except Exception:
                continue
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            crop = rgb[y1:y2, x1:x2]
            boxes_out.append([x1, y1, x2, y2])
            crops.append(crop)

        return boxes_out, crops

    def get_embedding(self, crop_rgb):
        img = Image.fromarray(crop_rgb).resize((160, 160))
        import torchvision.transforms as transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        x = transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.resnet(x)
        emb = emb.cpu().numpy().reshape(1, -1)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return emb.reshape(-1)

    def match_to_db(self, embedding):
        if embedding is None:
            return None, 0.0
        if len(self.db['embeddings']) == 0:
            return None, 0.0
        saved = np.vstack(self.db['embeddings'])
        sims = compute_cosine_sim(embedding.reshape(1, -1), saved)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= SIMILARITY_THRESHOLD:
            return self.db['ids'][best_idx], best_sim
        return None, best_sim

    def add_to_db(self, embedding, face_img=None):  # Added face_img param for age/gender prediction
        # Double-check for existing match before saving (to prevent duplicates)
        existing_pid, existing_sim = self.match_to_db(embedding)
        if existing_pid is not None and existing_sim >= SIMILARITY_THRESHOLD:
            print(f"[DB] Embedding matches existing person {existing_pid} (sim: {existing_sim:.2f}), skipping save.")
            return existing_pid  # Return existing PID to treat as match
        pid = f"person_{len(self.db['ids']) + 1}"
        self.db['ids'].append(pid)
        self.db['embeddings'].append(embedding)
        
        # Predict age and gender from face_img if provided
        age, gender = None, None
        if face_img is not None:
            try:
                age, gender = predict_age_gender(face_img)
            except Exception as e:
                print(f"Error predicting age/gender: {e}")
        meta = {
            'added': datetime.now().isoformat(),
            'age': age,
            'gender': gender
        }
        self.db['meta'].append(meta)
        save_db(self.db)
        log_attendance(pid)
        # Save to MongoDB with duplicate handling
        try:
            save_person(person_id=pid, age=age, gender=gender, embedding=embedding)
            print(f"[MongoDB] Saved new person {pid}")
        except Exception as e:
            error_msg = str(e)
            if "duplicate key error" in error_msg.lower():
                print(f"[MongoDB] Duplicate detected for {pid}, treating as existing match.")
                # Find existing PID in MongoDB (assuming it's the same)
                # For simplicity, return the generated PID if it matches; otherwise, skip
                # In a full fix, query MongoDB for the actual existing PID
                return pid  # Assume it's the same and proceed as match
            else:
                print(f"[MongoDB] Failed to save person {pid}: {e}")
        return pid

    def init_tracker_for_track(self, track, frame):
        try:
            track.tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            track.tracker = cv2.TrackerCSRT_create()
        track.tracker.init(frame, (int(track.bbox[0]), int(track.bbox[1]), int(track.bbox[2] - track.bbox[0]), int(track.bbox[3] - track.bbox[1])))

    def update_tracks(self, boxes, embeddings, matches=None):
        detections_centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes]
        assigned_tracks = set()
        assigned_dets = set()

        if len(self.tracks) == 0:
            for i, bbox in enumerate(boxes):
                emb = embeddings[i] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[i][0] is not None:
                    pid, sim = matches[i]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1
            return

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]
        if len(track_centroids) == 0:
            for i, bbox in enumerate(boxes):
                emb = embeddings[i] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[i][0] is not None:
                    pid, sim = matches[i]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1
            return

        D = np.zeros((len(track_centroids), len(detections_centroids)), dtype=np.float32)
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(detections_centroids):
                D[i, j] = np.linalg.norm(np.array(tc) - np.array(dc))

        while True:
            if D.size == 0:
                break
            i, j = np.unravel_index(np.argmin(D), D.shape)
            if D[i, j] > 100:
                break
            track_id = track_ids[i]
            if track_id in assigned_tracks or j in assigned_dets:
                D[i, j] = 1e6
                continue
            emb = embeddings[j] if embeddings else None
            self.tracks[track_id].update(boxes[j], self.frame_index, embedding=emb, reinit_tracker=True)
            if matches and matches[j][0] is not None:
                pid, sim = matches[j]
                self.tracks[track_id].saved = True
                self.tracks[track_id].note = f"matched:{pid}"
                log_attendance(pid)
            assigned_tracks.add(track_id)
            assigned_dets.add(j)
            D[i, :] = 1e6
            D[:, j] = 1e6

        for j, bbox in enumerate(boxes):
            if j not in assigned_dets:
                emb = embeddings[j] if embeddings else None
                t = Track(self.next_track_id, bbox, self.frame_index, embedding=emb)
                if matches and matches[j][0] is not None:
                    pid, sim = matches[j]
                    t.saved = True
                    t.note = f"matched:{pid}"
                    log_attendance(pid)
                self.tracks[self.next_track_id] = t
                self.next_track_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid].mark_missed()
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missed_frames > TRACKER_MAX_DISAPPEAR]
        for tid in to_delete:
            del self.tracks[tid]

    def step_on_frame(self, frame):
        events = []
        self.frame_index += 1

        is_detection_frame = (self.frame_index % DETECTION_EVERY_N_FRAMES) == 0

        if not is_detection_frame:
            for tid, tr in list(self.tracks.items()):
                if tr.tracker is not None:
                    success, box = tr.tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in box]
                        if w > 20 and h > 20 and w < frame.shape[1] * 0.8 and h < frame.shape[0] * 0.8:
                            tr.update_bbox_only([x, y, x + w, y + h])
                        else:
                            tr.mark_missed()
                    else:
                        tr.mark_missed()

        boxes, crops = [], []
        if is_detection_frame:
            try:
                boxes, crops = self.detect_faces(frame)
            except Exception:
                boxes, crops = [], []

            embeddings = []
            for crop in crops:
                try:
                    embeddings.append(self.get_embedding(crop))
                except Exception:
                    embeddings.append(None)

            if len(embeddings) != len(boxes):
                embeddings = [None] * len(boxes)

            self.update_tracks(boxes, embeddings)

            for tid, tr in self.tracks.items():
                try:
                    self.init_tracker_for_track(tr, frame)
                except Exception:
                    tr.tracker = None

        for tid, tr in list(self.tracks.items()):
            if not tr.is_visible:
                continue

            if tr.embedding is None:
                # Validate bbox before unpacking
                if not (isinstance(tr.bbox, (list, tuple)) and len(tr.bbox) == 4 and all(isinstance(v, (int, float)) for v in tr.bbox)):
                    continue  # Skip if bbox is invalid
                x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                h, w = frame.shape[:2]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
                    try:
                        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        tr.embedding = self.get_embedding(crop)
                    except Exception:
                        tr.embedding = None

            if tr.embedding is not None and not tr.saved:
                pid, sim = self.match_to_db(tr.embedding)
                if pid is not None:
                    tr.saved = True
                    tr.note = f"matched:{pid}"
                    log_attendance(pid)
                    events.append(('matched', pid, sim))
                else:
                    if tr.continuous_visible_time >= MIN_SECONDS_TO_SAVE:
                        # Crop face for age/gender prediction
                        if not (isinstance(tr.bbox, (list, tuple)) and len(tr.bbox) == 4 and all(isinstance(v, (int, float)) for v in tr.bbox)):
                            continue  # Skip if bbox is invalid
                        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
                        h, w = frame.shape[:2]
                        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
                        face_img = None
                        if x2 - x1 > 20 and y2 - y1 > 20:
                            face_img = frame[y1:y2, x1:x2]
                        new_pid = self.add_to_db(tr.embedding, face_img)  # Pass face_img for prediction
                        if new_pid in self.db['ids']:  # If saved successfully or matched
                            tr.saved = True
                            tr.note = f"matched:{new_pid}" if new_pid != f"person_{len(self.db['ids'])}" else f"new_saved:{new_pid}"  # Adjust note for matches
                            events.append(('matched', new_pid) if "matched" in tr.note else ('new_saved', new_pid))
                        else:
                            # If save failed and no match, reset for retry
                            tr.embedding = None

        annotated = frame.copy()
        for tid, tr in self.tracks.items():
            show_box = False
            if tr.saved:
                show_box = True
            elif tr.is_visible and tr.continuous_visible_time >= 3.0:
                show_box = True

            if show_box:
                if tr.note.startswith("matched:") or tr.note.startswith("new_saved:"):
                    identifier = tr.note.split(":")[1]
                    if identifier not in self.logged_visuals:
                        if identifier in self.db['ids']:
                            idx = self.db['ids'].index(identifier)
                            meta = self.db['meta'][idx] if idx < len(self.db['meta']) else {}
                            age = meta.get('age', '')
                            gender = meta.get('gender', '')
                        else:
                            age, gender = "", ""
                        log_visual_appearance(identifier=identifier, meta={"age": age, "gender": gender})
                        self.logged_visuals.add(identifier)

            if not show_box:
                continue

            x1, y1, x2, y2 = [int(v) for v in tr.bbox]
            color = BOX_COLOR if tr.saved else UNSAVED_COLOR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Track:{tid}"
            display_pid, display_age, display_gender = "", "", ""
            if tr.note.startswith("matched:") or tr.note.startswith("new_saved:"):
                pid = tr.note.split(":")[1]
                if pid in self.db['ids']:
                    idx = self.db['ids'].index(pid)
                    display_pid = pid  # Use pid directly
                    meta = self.db['meta'][idx] if idx < len(self.db['meta']) else {}
                    display_age = meta.get('age', '')
                    display_gender = meta.get('gender', '')

            if display_pid:
                info_line1 = f"{display_pid}"
                info_line2 = f"Age: {display_age} | Gender: {display_gender}" if display_age or display_gender else ""
            else:
                info_line1 = label
                info_line2 = f"{int(tr.continuous_visible_time)}s"

            cv2.putText(annotated, info_line1, (x1, max(y1 - 10, 0)), FONT, 0.6, TEXT_COLOR, 2)
            if info_line2:
                cv2.putText(annotated, info_line2, (x1, y2 + 20), FONT, 0.5, TEXT_COLOR, 1)

        db_count = len(self.db['ids'])
        cv2.putText(annotated, f"Total Saved Persons: {db_count}", (10, 25), FONT, 0.7, (0, 255, 255), 2)

        return annotated, events



