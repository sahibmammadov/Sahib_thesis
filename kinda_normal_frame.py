import cv2
import dlib
import numpy as np
from deepface import DeepFace
import logging
import face_recognition
import pickle
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\sahib\Downloads\shape_predictor_68_face_landmarks.dat")

logging.basicConfig(level=logging.INFO)
smth = 9
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

AGE_BUCKETS = [2, 6, 12, 17, 25, 36, 46, 60]  
GENDER_LABELS = ["Male", "Female"]

try:
    with open("known_face.pkl", "rb") as f:
        known_face_encodings, known_face_ids = pickle.load(f)
except FileNotFoundError:
    known_face_encodings, known_face_ids = [], []

next_id = len(known_face_ids) + 1
user_dict = {}

def classify_age(age_value):
    if age_value <= 12:
        return "Child"
    elif age_value <= 17:
        return "Teenager"
    return "Adult"

def detect_gender_age(face_roi):
    face_resized = cv2.resize(face_roi, (256, 256))  
    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LABELS[np.argmax(gender_preds)]

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_value = sum(np.array(AGE_BUCKETS) * age_preds.flatten()) 
    age_category = classify_age(age_value)

    return gender, age_category, int(age_value)  

def recognize_or_add_faces(face_encoding):
    global next_id
    if known_face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.51:  
            return known_face_ids[best_match_index]
    
    new_id = next_id
    known_face_encodings.append(face_encoding)
    known_face_ids.append(new_id)
    next_id += 1
    return new_id

def get_eye_region(landmarks, eye_points):
    return np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in eye_points], np.int32)

def get_eye_center(eye_region):
    x = np.mean(eye_region[:, 0])
    y = np.mean(eye_region[:, 1])
    return int(x), int(y)

def get_pupil_position(eye_frame):
    gray = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            return int(M["m10"] / M["m00"])  
    return None

def get_gaze_and_head_direction(landmarks, frame):

    left_eye_region = get_eye_region(landmarks, [36, 37, 38, 39, 40, 41])
    right_eye_region = get_eye_region(landmarks, [42, 43, 44, 45, 46, 47])

    left_x, left_y = get_eye_center(left_eye_region)
    right_x, right_y = get_eye_center(right_eye_region)

    eye_width, eye_height = 50, 30
    left_eye_frame = frame[left_y - eye_height//2:left_y + eye_height//2, left_x - eye_width//2:left_x + eye_width//2]
    right_eye_frame = frame[right_y - eye_height//2:right_y + eye_height//2, right_x - eye_width//2:right_x + eye_width//2]

    if left_eye_frame.shape[0] == 0 or left_eye_frame.shape[1] == 0 or right_eye_frame.shape[0] == 0 or right_eye_frame.shape[1] == 0:
        return "Unknown"

    left_pupil_x = get_pupil_position(left_eye_frame)
    right_pupil_x = get_pupil_position(right_eye_frame)

    nose_x = landmarks.part(30).x

    left_face_x = landmarks.part(0).x  
    right_face_x = landmarks.part(16).x  
    face_center_x = (left_face_x + right_face_x) // 2

    head_direction = "Center"
    if nose_x < face_center_x - 10:  
        head_direction = "Left"
    elif nose_x > face_center_x + 10:  
        head_direction = "Right"

    if left_pupil_x is None or right_pupil_x is None:
        return "Unknown"

    gaze_direction = "Center"
    if left_pupil_x < eye_width * 0.3 and right_pupil_x < eye_width * 0.3:
        gaze_direction = "Left"
    elif left_pupil_x > eye_width * 0.7 and right_pupil_x > eye_width * 0.7:
        gaze_direction = "Right"

    if head_direction == "Center" and gaze_direction == "Center":
        return "Looking at Camera"
    else:
        return "Looking Side"
    
def all():
    global frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    face_net.setInput(blob)
    detections = face_net.forward()
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        margin = 15  
        top = max(0, top - margin)
        bottom = min(h, bottom + margin)
        left = max(0, left - margin)
        right = min(w, right + margin)

        face_roi = frame[top:bottom, left:right]

        if face_roi.shape[0] > 10 and face_roi.shape[1] > 10:
            try:
                analysis = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
                analysis = analysis[0] if isinstance(analysis, list) else analysis

                matched_user = recognize_or_add_faces(face_encoding)

                if matched_user in user_dict:
                    emotions = analysis.get('emotion', {})
                    relevant_emotions = {key: emotions.get(key, 0) for key in ["happy", "sad", "angry", "surprise", "neutral"]}
                    dominant_emotion = max(relevant_emotions, key=relevant_emotions.get)
                    emotion_confidence = relevant_emotions[dominant_emotion]

                    user_dict[matched_user][2] = dominant_emotion  
                    gender = user_dict[matched_user][0]
                    age_category = user_dict[matched_user][1]
                    print(user_dict)
                    print(matched_user)
                else:
                    print(matched_user)
                    gender, age_category, age_value = detect_gender_age(face_roi)
                    emotions = analysis.get('emotion', {})
                    relevant_emotions = {key: emotions.get(key, 0) for key in ["happy", "sad", "angry", "surprise", "neutral"]}

                    dominant_emotion = max(relevant_emotions, key=relevant_emotions.get)
                    emotion_confidence = relevant_emotions[dominant_emotion]

                    user_dict[matched_user] = [gender, age_category, dominant_emotion]
                    print(user_dict)

                emotion_colors = {"happy": (0, 255, 0),"sad": (255, 0, 0), "angry": (0, 0, 255), "surprise": (255, 255, 0) }

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                gender_color = (0, 0, 255) if gender == "Male" else (255, 0, 255)
                cv2.putText(frame, f"Gender: {gender}", (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)

                cv2.putText(frame, f"Age: {age_category}", (left, bottom + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                emotion_color = emotion_colors.get(dominant_emotion, (255, 255, 255))
                cv2.putText(frame, f"Emotion: {dominant_emotion} ({emotion_confidence:.2f})", (left, bottom + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)

            except Exception as e:
                logging.error(f"Error in Face Processing: {e}")

#cap = cv2.VideoCapture(r"C:\Users\sahib\Desktop\thesis\video_other.mp4")
cap = cv2.VideoCapture(0)

while True:
    global frame
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    

    for face in faces:
        landmarks = predictor(gray, face)
        gaze_status = get_gaze_and_head_direction(landmarks, frame)
        if gaze_status == "Looking at Camera":
            smth += 1
            if smth == 10:
                smth = 0
                all()
        cv2.putText(frame, gaze_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for n in range(36, 48):  # Draw eye landmarks
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow("Gaze Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
