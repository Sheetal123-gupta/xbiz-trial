import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from collections import deque

# ---------- Config ----------
YOLO_MODEL_PATH = "yolov8n.pt"
PHONE_CLASS_NAMES = {"cell phone", "mobile phone", "phone"}
CONF_THRESH = 0.25
SPOOF_FRAMES_THRESHOLD = 3
MIN_PHONE_SIZE = 80

# Load models
model = YOLO(YOLO_MODEL_PATH)
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=2)

cap = cv2.VideoCapture(0)

# ring buffer for spoof smoothing
recent_spoof = deque(maxlen=SPOOF_FRAMES_THRESHOLD)
spoof_counter = 0

def mp_face_landmarks_bbox(landmarks, w, h, offset_x=0, offset_y=0):
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x_min = int(min(xs) * w) + offset_x
    x_max = int(max(xs) * w) + offset_x
    y_min = int(min(ys) * h) + offset_y
    y_max = int(max(ys) * h) + offset_y
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    return max(x_min - pad_x, 0), max(y_min - pad_y, 0), x_max + pad_x, y_max + pad_y

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # Run YOLO
    results = model(frame, conf=CONF_THRESH, verbose=False)

    phone_boxes = []
    if len(results) > 0:
        res0 = results[0]
        for box in res0.boxes:
            try:
                cls_id = int(box.cls.item())
            except Exception:
                cls_id = int(box.cls.cpu().numpy().astype(int).item())
            cls_name = model.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

            if any(pk in cls_name.lower() for pk in PHONE_CLASS_NAMES):
                if (x2 - x1) >= MIN_PHONE_SIZE and (y2 - y1) >= MIN_PHONE_SIZE:
                    phone_boxes.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 100, 20), 2)
                cv2.putText(frame, f"PHONE: {cls_name}", (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 20), 2)

    frame_spoof_detected = False

    # Check for faces in full frame
    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    full_face_results = mp_face.process(rgb_full)
    real_face_bboxes = []
    if full_face_results.multi_face_landmarks:
        for face_landmarks in full_face_results.multi_face_landmarks:
            fx1, fy1, fx2, fy2 = mp_face_landmarks_bbox(face_landmarks, w_frame, h_frame)
            real_face_bboxes.append((fx1, fy1, fx2, fy2))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 200, 0), 2)
            cv2.putText(frame, "FACE (candidate REAL)", (fx1, max(10, fy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

    # Check inside phone ROIs
    for (px1, py1, px2, py2) in phone_boxes:
        pad = 6
        rx1, ry1 = max(px1 - pad, 0), max(py1 - pad, 0)
        rx2, ry2 = min(px2 + pad, w_frame - 1), min(py2 + pad, h_frame - 1)
        phone_roi = frame[ry1:ry2, rx1:rx2]
        if phone_roi.size == 0:
            continue

        rgb_roi = cv2.cvtColor(phone_roi, cv2.COLOR_BGR2RGB)
        roi_results = mp_face.process(rgb_roi)

        if roi_results.multi_face_landmarks:
            frame_spoof_detected = True
            for face_lms in roi_results.multi_face_landmarks:
                r_w = rx2 - rx1
                r_h = ry2 - ry1
                bx1, by1, bx2, by2 = mp_face_landmarks_bbox(face_lms, r_w, r_h, rx1, ry1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(frame, "SPOOF (face inside phone)", (bx1, max(10, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 200, 100), 1)

    # Update spoof buffer
    recent_spoof.append(frame_spoof_detected)
    if sum(recent_spoof) >= SPOOF_FRAMES_THRESHOLD:
        spoof_counter = min(spoof_counter + 1, SPOOF_FRAMES_THRESHOLD)
    else:
        spoof_counter = max(0, spoof_counter - 1)

    # Display label
    if spoof_counter >= SPOOF_FRAMES_THRESHOLD:
        final_label = "SPOOF DETECTED"
        color = (0, 0, 255)
    elif len(real_face_bboxes) > 0:
        final_label = "REAL"
        color = (0, 200, 0)
    else:
        final_label = "NO FACE"
        color = (200, 200, 200)

    cv2.putText(frame, final_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow("YOLO + MediaPipe Anti-Spoof", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



'''
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from collections import deque
from math import hypot # Import for distance calculation

# ---------- Config ----------
YOLO_MODEL_PATH = "C:\\Users\\ASUS\\Music\\xbiz-trial\\object_detection\\yolov8n.pt" 
PHONE_CLASS_NAMES = {"cell phone", "mobile phone", "phone"} 
CONF_THRESH = 0.25
SPOOF_FRAMES_THRESHOLD = 3 
MIN_PHONE_SIZE = 80 
MIN_MOVEMENT_THRESHOLD = 5 
MAX_STATIC_FRAMES = 10 

model = YOLO(YOLO_MODEL_PATH)
# Increased max_num_faces to potentially find both the person holding the phone and the face on the phone
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=4) 

cap = cv2.VideoCapture(0)
recent_spoof = deque(maxlen=SPOOF_FRAMES_THRESHOLD)
spoof_counter = 0

# NEW GLOBAL VARIABLES for motion tracking
last_spoof_center = None 
static_frame_count = 0 

def mp_face_landmarks_bbox(landmarks, w, h, offset_x=0, offset_y=0):
# ... (your existing mp_face_landmarks_bbox function)
    xs = [lm.x for lm in landmarks.landmark]
    ys = [lm.y for lm in landmarks.landmark]
    x_min = int(min(xs) * w) + offset_x
    x_max = int(max(xs) * w) + offset_x
    y_min = int(min(ys) * h) + offset_y
    y_max = int(max(ys) * h) + offset_y
    # small padding
    pad_x = int((x_max - x_min) * 0.1)
    pad_y = int((y_max - y_min) * 0.1)
    
    # Calculate center for motion tracking
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    return max(x_min - pad_x, 0), max(y_min - pad_y, 0), x_max + pad_x, y_max + pad_y, (center_x, center_y) # <-- ADDED CENTER

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]

    # Run YOLO (existing code)
    results = model(frame, conf=CONF_THRESH, verbose=False) 
    phone_boxes = []
    # ... (collect phone_boxes and draw basic phone box)
    if len(results) > 0:
        res0 = results[0]
        for box in res0.boxes:
            # ... (your existing logic to extract cls_id, cls_name, and xyxy)
            try:
                cls_id = int(box.cls.item())
            except Exception:
                cls_id = int(box.cls.cpu().numpy().astype(int).item())
            cls_name = model.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy() 
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

            if any(pk in cls_name.lower() for pk in PHONE_CLASS_NAMES):
                if (x2 - x1) >= MIN_PHONE_SIZE and (y2 - y1) >= MIN_PHONE_SIZE:
                    phone_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 100, 20), 2)
                    cv2.putText(frame, f"PHONE: {cls_name}", (x1, max(10, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 20), 2)
    # --------------------------------

    frame_spoof_detected = False
    current_spoof_center = None 
    
    # 1) Check for face landmarks in the whole frame (for REAL face)
    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    full_face_results = mp_face.process(rgb_full)

    real_face_bboxes = []
    if full_face_results.multi_face_landmarks:
        for face_landmarks in full_face_results.multi_face_landmarks:
            # Note: Now mp_face_landmarks_bbox returns 5 values (including center)
            fx1, fy1, fx2, fy2, _ = mp_face_landmarks_bbox(face_landmarks, w_frame, h_frame, 0, 0) 
            real_face_bboxes.append((fx1, fy1, fx2, fy2))
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 200, 0), 2)
            cv2.putText(frame, "FACE (candidate REAL)", (fx1, max(10, fy1 - 8)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    # --------------------------------

    # 2) For each detected phone, check inside phone ROI for face landmarks
    for (px1, py1, px2, py2) in phone_boxes:
        pad = 6
        rx1, ry1 = max(px1 - pad, 0), max(py1 - pad, 0)
        rx2, ry2 = min(px2 + pad, w_frame - 1), min(py2 + pad, h_frame - 1)
        phone_roi = frame[ry1:ry2, rx1:rx2]
        if phone_roi.size == 0:
            continue

        rgb_roi = cv2.cvtColor(phone_roi, cv2.COLOR_BGR2RGB)
        roi_results = mp_face.process(rgb_roi)

        if roi_results.multi_face_landmarks:
            frame_spoof_detected = True # Found a face inside the phone
            # We only track the first face found on a phone for simplicity
            face_lms = roi_results.multi_face_landmarks[0]
            r_w = rx2 - rx1
            r_h = ry2 - ry1
            
            # Get the bbox and center of the face inside the phone
            bx1, by1, bx2, by2, current_spoof_center = mp_face_landmarks_bbox(
                face_lms, r_w, r_h, offset_x=rx1, offset_y=ry1
            )
            
            # draw a clear red bbox for spoof face inside phone
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
            cv2.putText(frame, "SPOOF (face inside phone)", (bx1, max(10, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            break # Once we detect one spoof face, we break out of the phone loop
        else:
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (100, 200, 100), 1)

    # 3) Temporal smoothing (debounce) and NEW LIVENESS/MOTION CHECK
    
    is_video_spoof = False
    
    if current_spoof_center:
        # We detected a face inside a phone
        
        # Check for movement from the last frame (for Liveness/Static detection)
        if last_spoof_center:
            # Calculate pixel distance moved
            movement_distance = hypot(current_spoof_center[0] - last_spoof_center[0], 
                                      current_spoof_center[1] - last_spoof_center[1])
            
            if movement_distance > MIN_MOVEMENT_THRESHOLD:
                # Face on screen is moving (either a real face or a video)
                static_frame_count = 0 
                is_video_spoof = True # Assume video spoof if moving on the phone
                cv2.putText(frame, "MOTION DETECTED", (current_spoof_center[0] - 50, current_spoof_center[1] + (by2-by1)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
            else:
                # Face on screen is static or moving very little
                static_frame_count += 1
                if static_frame_count >= MAX_STATIC_FRAMES:
                     # This could be a static photo or a paused video
                     # For the purpose of "video or uploaded", a static photo is also a spoof
                     is_video_spoof = True 
                     cv2.putText(frame, "STATIC SPOOF", (current_spoof_center[0] - 50, current_spoof_center[1] + (by2-by1)//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        last_spoof_center = current_spoof_center
        
        # If a face is detected on a phone, and it's either moving or static, it's a spoof.
        # We rely on your base logic: Face in Phone = Spoof
        
        recent_spoof.append(True) # Face in phone is always a spoof
    else:
        # No face in phone detected in this frame
        last_spoof_center = None
        static_frame_count = 0
        recent_spoof.append(False) 


    # Display final top-left status with smoothing (same as existing logic)
    true_count = sum(1 for v in recent_spoof if v)
    if true_count >= SPOOF_FRAMES_THRESHOLD:
        spoof_counter = SPOOF_FRAMES_THRESHOLD
    else:
        spoof_counter = max(0, spoof_counter - 1)

    if spoof_counter >= SPOOF_FRAMES_THRESHOLD:
        final_label = "SPOOF DETECTED (Face on Phone)"
        color = (0, 0, 255)
    else:
        if len(real_face_bboxes) > 0:
            final_label = "REAL"
            color = (0, 200, 0)
        else:
            final_label = "NO FACE"
            color = (200, 200, 200)

    cv2.putText(frame, final_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("YOLO + MediaPipe Anti-Spoof (phone-inside-face check)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''