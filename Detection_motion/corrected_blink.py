'''
import cv2
import mediapipe as mp
import os
import math
import time

# Create folder to save blink frames
SAVE_DIR = "saved_blink"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (from MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # around left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # around right eye

def euclidean_dist(pt1, pt2):
    return math.dist(pt1, pt2)

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    # Extract coordinates
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    # EAR calculation
    vert1 = euclidean_dist(coords[1], coords[5])
    vert2 = euclidean_dist(coords[2], coords[4])
    horiz = euclidean_dist(coords[0], coords[3])
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

# Blink detection thresholds
EAR_THRESHOLD = 0.21  # tweak if too sensitive
BLINK_CONSEC_FRAMES = 2

blink_counter = 0
total_blinks = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Compute EAR for both eyes
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    total_blinks += 1
                    # Save the blink frame
                    filename = os.path.join(SAVE_DIR, f"blink_{int(time.time())}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Blink detected! Saved: {filename}")
                blink_counter = 0

            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            )

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''

import cv2
import mediapipe as mp
import math

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (from MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # left eye landmarks
RIGHT_EYE = [362, 385, 387, 263, 373, 380] # right eye landmarks

def euclidean_dist(pt1, pt2):
    return math.dist(pt1, pt2)

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    coords = [(int(landmarks[i].x * image_w), int(landmarks[i].y * image_h)) for i in eye_indices]
    vert1 = euclidean_dist(coords[1], coords[5])
    vert2 = euclidean_dist(coords[2], coords[4])
    horiz = euclidean_dist(coords[0], coords[3])
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

# Blink detection thresholds
EAR_THRESHOLD = 0.21       # adjust if too sensitive
BLINK_CONSEC_FRAMES = 2    # consecutive frames required

blink_counter = 0
total_blinks = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    total_blinks += 1
                    print(f"Blink detected! Total: {total_blinks}")
                blink_counter = 0

            # Optional: draw mesh
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            )

    # Show blink counter on frame
    cv2.putText(frame, f"Blinks: {total_blinks}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
