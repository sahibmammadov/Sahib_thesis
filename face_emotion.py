import cv2
from deepface import DeepFace

def preprocess_face(image):
    #making gray and equalize to detect better
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)  

def normalize_emotions(emotion_data):
    #choosing emotions
    emotions = ['happy', 'neutral', 'sad', 'angry', 'surprise']
    choosed_emotions = {key: emotion_data.get(key, 0) for key in emotions}
    #normalize emotions
    total = sum(choosed_emotions.values())
    if total > 0:
        return {key: value / total for key, value in choosed_emotions.items()}
    return choosed_emotions

def start():
    
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break
        mirrored_frame = cv2.flip(frame, 1)

        #making gray for face detection
        gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
        
        #for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray,1.1, 5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            # calculating the face region
            face_rg = mirrored_frame[y:y+h, x:x+w]
            if face_rg.size == 0:
                continue
            preprocessed_face = preprocess_face(face_rg)

            try:
                #analyze emotions
                analysis = DeepFace.analyze(preprocessed_face, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
                emotion_data = analysis['emotion'] if isinstance(analysis, dict) else analysis[0]['emotion']

                #choosing emotions
                emotions = normalize_emotions(emotion_data)

                #finding the most probable emotion and its rate
                pr_emotion = max(emotions, key=emotions.get)
                rate = emotions[pr_emotion]
                emotion_text = ', '.join(f"{key}: {value:.2f}" for key, value in emotions.items())

                #box around the face
                cv2.rectangle(mirrored_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(mirrored_frame, f"Dominant: {pr_emotion} ({rate:.2f})", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
                cv2.putText(mirrored_frame, emotion_text, (x, y+h+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            #exsept shows error of face detection
            except Exception as e:
                print(f"Emotion detection failed: {e}")

        cv2.imshow('Precise Emotion Detection', mirrored_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start()
