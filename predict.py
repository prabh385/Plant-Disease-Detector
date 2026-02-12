import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("Model\plant_disease_model.h5")

CONFIDENCE_THRESHOLD = 85   # you can change to 80 if needed


class_names = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Blueberry___healthy', 
    'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
    'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 
    'Corn_(maize)___healthy', 
    'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 
    'Peach___Bacterial_spot', 
    'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Raspberry___healthy', 
    'Soybean___healthy', 
    'Squash___Powdery_mildew', 
    'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

cap = cv2.VideoCapture(0)  # 0 = default webcam

best_label = ""
best_confidence = 0
freeze_frame = None


while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (96,96))
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    label = class_names[class_id]

    # Track highest confidence seen so far
    if confidence > best_confidence:
        best_confidence = confidence
        best_label = label
        freeze_frame = frame.copy()

    # If confidence crosses threshold → stop
    if best_confidence >= CONFIDENCE_THRESHOLD:
        cv2.putText(freeze_frame,
                    f"FINAL: {best_label} ({best_confidence:.2f}%)",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

        cv2.imshow("Plant Disease Detection", freeze_frame)
        cv2.waitKey(0)
        break

    # Show live scanning
    cv2.putText(frame,
                f"Scanning... {label} ({confidence:.2f}%)",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2)

    cv2.imshow("Plant Disease Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
