import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# --- CONFIGURATION ---
YOLO_MODEL_PATH = r"C:\Users\ASUS\Music\xbiz-trial\object_detection\yolov8n.pt"
CLASSIFIER_PATH = r"C:\Users\ASUS\Downloads\sign_shape_classifier"  # SavedModel folder
DEVICE_CLASSES = ['cell phone', 'laptop', 'tv', 'monitor']
IMAGE_SIZE = (128, 128)
CLASS_NAMES_CLASSIFIER = ["Signature", "Shape", "Background"]
THRESHOLD = 0.5

# --- LOAD MODELS ---
print("[INFO] Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
classifier_model = tf.keras.models.load_model(CLASSIFIER_PATH)
print("[READY] Models loaded successfully.")

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not available.")

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Detect objects
    results = yolo_model(frame, conf=0.5, verbose=False)
    detected_devices = []
    detections = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            buffer = 15
            x1, y1 = max(0, x1 - buffer), max(0, y1 - buffer)
            x2, y2 = min(width, x2 + buffer), min(height, y2 + buffer)

            if label in DEVICE_CLASSES:
                detected_devices.append((x1, y1, x2, y2))
                color = (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                detections.append((x1, y1, x2, y2, label))

    # Classify non-device regions
    for (x1, y1, x2, y2, _) in detections:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        img_resized = cv2.resize(crop, IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = classifier_model.predict(img_array, verbose=0)[0]
        pred_class = np.argmax(preds)
        confidence = np.max(preds)
        predicted_label = CLASS_NAMES_CLASSIFIER[pred_class]

        if confidence < THRESHOLD:
            text, color = "Unknown", (0, 0, 255)
        else:
            # Determine context (on device or paper)
            on_device = any(
                x1 < dx2 and x2 > dx1 and y1 < dy2 and y2 > dy1
                for (dx1, dy1, dx2, dy2) in detected_devices
            )

            if predicted_label in ["Signature", "Shape"]:
                location = "on device" if on_device else "on paper"
                text = f"{predicted_label} {location}"
                color = (0, 165, 255) if on_device else (0, 255, 0)
            else:
                text, color = predicted_label, (200, 200, 200)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    device_msg = f"Devices detected: {len(detected_devices)}"
    cv2.putText(frame, device_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Hybrid Sign/Shape Context Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Session ended.")

