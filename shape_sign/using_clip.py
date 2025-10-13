'''
import torch
import clip
from PIL import Image
import cv2

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# --- Prompts for classification ---
prompts = [
    "a shape on Device",
    "a signature on paper",
    "a shape on paper",
    "a signature on a Device"
]

# Tokenize prompts and precompute text features
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# --- Open webcam ---
cap = cv2.VideoCapture(0)  # 0 is default camera
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("[INFO] Starting live webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Preprocess and encode with CLIP
    image_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_idx = similarity.argmax().item()
        confidence = similarity[0][best_idx].item()
        label = prompts[best_idx]

    # Display label on frame
    cv2.putText(frame, f"{label} ({confidence*100:.1f}%)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("CLIP Webcam Detector", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import torch
import clip
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# --- CLIP prompts ---
prompts = [
    "a shape on Device",
    "a signature on paper",
    "a shape on paper",
    "a signature on a Device"
]
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# --- Load YOLO model trained for devices/paper detection ---
yolo_model = YOLO("C:\\Users\\ASUS\\Music\\xbiz-trial\\object_detection\\yolov8n.pt")
DEVICE_CLASSES = ['cell phone', 'laptop']
PAPER_CLASSES = ['paper']

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # --- YOLO detection ---
    results = yolo_model(frame, conf=0.5, verbose=False)

    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            label_detected = yolo_model.names[class_id]

            if label_detected in DEVICE_CLASSES + PAPER_CLASSES:
                # Crop detected object
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # --- CLIP classification ---
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                image_tensor = preprocess(pil_crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    best_idx = similarity.argmax().item()
                    confidence = similarity[0][best_idx].item()
                    final_label = f"{prompts[best_idx]} ({confidence*100:.1f}%)"

                # Draw bounding box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, final_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("YOLO + CLIP Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

import os

# Path to your folder
folder_path = r"C:\Users\ASUS\Downloads\CLIP_Labeled_Output2"  #  change this as needed

# Count files
file_count = sum(
    1 for entry in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, entry))
)

print(f"Total files in folder: {file_count}")
