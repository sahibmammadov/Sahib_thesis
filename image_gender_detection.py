import cv2
from deepface import DeepFace
import logging

logging.basicConfig(level=logging.INFO)

def detect_gender_from_image(image_path):
    try:
        image = cv2.imread('women.jpg')
        if image is None:
            logging.error("Could not read the image. Please check the file path.")
            return

        max_dimension = 800  
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

        if len(faces) == 0:
            logging.info("No faces detected in the image.")
            return

        for (x, y, w, h) in faces:
            face_rg = image[y:y+h, x:x+w]

            if face_rg.size == 0:
                logging.warning("Empty face region detected.")
                continue

            analysis = DeepFace.analyze(face_rg, actions=['gender'], enforce_detection=False, detector_backend='mtcnn')

            if isinstance(analysis, list) and len(analysis) > 0:
                analysis = analysis[0]  

            dominant_gender = analysis.get('dominant_gender', 'Unknown')
            confidence = analysis.get('gender_prediction', {})

            logging.info(f"Detected gender: {dominant_gender}, Confidence: {confidence}")
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, f"Gender: {dominant_gender}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Gender Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logging.error(f"Error during gender detection: {e}")

if __name__ == "__main__":
    detect_gender_from_image("women.jpg")
