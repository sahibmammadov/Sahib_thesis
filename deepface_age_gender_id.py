import cv2
from deepface import DeepFace
import logging
import numpy as np

global user_id
user_id = 1

logging.basicConfig(level=logging.INFO)
known_faces = {}
def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)  

def normalize_emotions(emotion_data):
    emotions = ['happy', 'neutral', 'sad', 'angry', 'surprise']
    choosed_emotions = {key: emotion_data.get(key, 0) for key in emotions}
    if sum(choosed_emotions.values()) < 0.1:
        choosed_emotions['neutral'] += 0.1
    total = sum(choosed_emotions.values())
    if total > 0:
        return {key: value / total for key, value in choosed_emotions.items()}
    return choosed_emotions

def find_matching_user(face_embedding):
    """Find a matching user based on face embedding."""
    for user, embedding in known_faces.items():
        distance = np.linalg.norm(embedding - np.array(face_embedding))
        if distance < 10: 
            return user
    return None
def start_camera_detection():
    global user_id
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture video.")
            break

        frame = cv2.flip(frame, 1)

        max_dimension = 800  
        height, width = frame.shape[:2]
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            face_rg = frame[y:y+h, x:x+w]

            if face_rg.size == 0:
                logging.warning("Empty face region detected.")
                continue

            face_rg = preprocess_face(face_rg)

            try:
                analysis = DeepFace.analyze(face_rg, actions=['emotion', 'gender', 'age'], enforce_detection=False, detector_backend='mtcnn')
                
                result = DeepFace.represent(frame, model_name="Facenet")
                face_embedding = result[0]['embedding']
                matched_user = find_matching_user(face_embedding)
                if isinstance(analysis, list) and len(analysis) > 0:
                        analysis = analysis[0] 
                if matched_user:
                        print(f"Welcome back, User {matched_user}!")
                        
                else:
                        print(f"Hello, User {user_id}!")
                        known_faces[user_id] = np.array(face_embedding)
                        user_id += 1
        
                        gender = analysis.get('dominant_gender', 'Unknown')
                        confidence = analysis.get('gender', {})
                        logging.info(f"Detected gender: {gender}, Confidence: {confidence}")
                        m_w = " ".join(gender)
                        gender = m_w.split()[0]  
                        age = max(1, min(99, int(analysis.get('age', 0)))) - 3 
                        

                emotions = normalize_emotions(analysis.get('emotion', {}))
                pr_emotion = max(emotions, key=emotions.get)
                rate = emotions[pr_emotion]
                emotion_text = ', '.join(f"{key}: {value:.2f}" for key, value in emotions.items())

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"Dominant: {pr_emotion} ({rate:.2f}) [{gender}]", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
                cv2.putText(frame, emotion_text, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, f"Age: {age}", (x, y+h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            except Exception as e:
                logging.error(f"Detection failed: {e}")

        cv2.imshow("Gender, Age, Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera_detection()