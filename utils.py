import pandas as pd
import cv2
import time
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------- Core Config --------------------------
AGE_PROTO = "age_gender_model/deploy_age.prototxt"
AGE_MODEL = "age_gender_model/age_net.caffemodel"
GENDER_PROTO = "age_gender_model/deploy_gender.prototxt"
GENDER_MODEL = "age_gender_model/gender_net.caffemodel"

# Load models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

AGE_LIST = ['(10-20)', '(20-30)', '(30-40)', '(40-50)', '(50-60)',
            '(60-100)']
GENDER_LIST = ['Male', 'Female']

# -------------------------- Predict Age & Gender --------------------------
def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]

    return age, gender

def compute_cosine_sim(a, b):
    return cosine_similarity(a, b)[0]

# -------------------------- Track with OpenCV tracker for smooth motion --------------------------
class Track:
    def __init__(self, track_id, bbox, frame_index, embedding=None):
        self.id = track_id
        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_seen = frame_index
        self.continuous_visible_time = 0.0
        self.last_update_time = time.time()
        self.saved = False
        self.embedding = embedding
        self.missed_frames = 0
        self.note = ''
        self.tracker = None
        self.is_visible = True

    def compute_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, bbox, frame_index, embedding=None, reinit_tracker=False):
        current_time = time.time()
        if self.is_visible and hasattr(self, 'last_update_time'):
            self.continuous_visible_time += (current_time - self.last_update_time)
        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_seen = frame_index
        self.last_update_time = current_time
        self.is_visible = True
        if embedding is not None:
            self.embedding = embedding
        self.missed_frames = 0
        if reinit_tracker:
            self.tracker = None

    def update_bbox_only(self, bbox):
        current_time = time.time()
        if self.is_visible and hasattr(self, 'last_update_time'):
            self.continuous_visible_time += (current_time - self.last_update_time)
        self.bbox = bbox
        self.centroid = self.compute_centroid(bbox)
        self.last_update_time = current_time
        self.is_visible = True

    def mark_missed(self):
        self.missed_frames += 1
        self.is_visible = False
        self.continuous_visible_time = 0.0