import cv2
import numpy as np
import os

BASE_DIR = r"C:\Users\sahib\Downloads" 

FACE_PROTO = os.path.join(BASE_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join(BASE_DIR, "deploy_age.prototxt")
AGE_MODEL = os.path.join(BASE_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(BASE_DIR, "deploy_gender.prototxt")
GENDER_MODEL = os.path.join(BASE_DIR, "gender_net.caffemodel")

print("[INFO] Loading models...")
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
print("[INFO] Models loaded successfully!")

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LABELS = ["Male", "Female"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            x, y, x1, y1 = max(0, x), max(0, y), min(w - 1, x1), min(h - 1, y1)
            face = frame[y:y1, x:x1]

            if face.shape[0] < 10 or face.shape[1] < 10:  
                continue

            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LABELS[gender_preds[0].argmax()]

            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = AGE_BUCKETS[age_preds[0].argmax()]

            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

    cv2.imshow("Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
