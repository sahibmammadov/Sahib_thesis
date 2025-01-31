import cv2
from deepface import DeepFace
import logging
import numpy as np
import face_recognition
import pickle
import os

logging.basicConfig(level=logging.INFO)

BASE_DIR = r"C:\Users\sahib\Downloads"
FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join(BASE_DIR, "deploy_age.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "deploy_gender.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")

face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-17", "18-25", "26-35", "36-45", "46-60", "60-100"]
GENDER_LABELS = ["Male", "Female"]

try:
    with open("known_face.pkl", "rb") as f:
        known_face_encodings, known_face_ids = pickle.load(f)
except FileNotFoundError:
    known_face_encodings, known_face_ids = [], []

next_id = len(known_face_ids) + 1
user_dict = {}

def classify_age(age_group):
    if age_group in ["0-2", "4-6", "8-12"]:
        return "Child"
    elif age_group in ["15-17"]:
        return "Teenager"
    return "Adult"

def detect_gender_age(face_roi):
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
    
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LABELS[np.argmax(gender_preds)]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_group = AGE_BUCKETS[np.argmax(age_preds)]
    age_category = classify_age(age_group)

    return gender, age_category

def recognize_or_add_faces(face_encoding):
    global next_id
    if known_face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.5:  
            return known_face_ids[best_match_index]
    
    new_id = next_id
    known_face_encodings.append(face_encoding)
    known_face_ids.append(new_id)
    next_id += 1
    return new_id

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    face_net.setInput(blob)
    detections = face_net.forward()
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_roi = frame[top:bottom, left:right]

        if face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
            try:
                analysis = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
                analysis = analysis[0] if isinstance(analysis, list) else analysis

                emotions = analysis.get('emotion', {})
                relevant_emotions = {key: emotions.get(key, 0) for key in ["happy", "sad", "angry", "surprise", "neutral"]}

                dominant_emotion = max(relevant_emotions, key=relevant_emotions.get)
                emotion_confidence = relevant_emotions[dominant_emotion]

                matched_user = recognize_or_add_faces(face_encoding)

                if matched_user in user_dict:
                    gender, age_category = user_dict[matched_user]
                else:
                    gender, age_category = detect_gender_age(face_roi)
                    user_dict[matched_user] = (gender, age_category)
                    print(user_dict)
                emotion_colors = {"happy": (0, 255, 0),"sad": (255, 0, 0),"angry": (0, 0, 255), "surprise": (255, 255, 0),}

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                gender_color = (0, 0, 255) if gender == "Male" else (255, 0, 255)
                cv2.putText(frame, f"Gender: {gender}", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)

                cv2.putText(frame, f"Age: {age_category}", (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                emotion_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
                cv2.putText(frame, f"Emotion: {dominant_emotion} ({emotion_confidence:.2f})", (left, bottom + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            except Exception as e:
                logging.error(f"Error in Face Processing: {e}")

    cv2.imshow("Age, Gender & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("known_face.pkl", "wb") as f:
    pickle.dump((known_face_encodings, known_face_ids), f)

video_capture.release()
cv2.destroyAllWindows()
