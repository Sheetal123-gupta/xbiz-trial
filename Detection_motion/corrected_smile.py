import cv2
import mediapipe as mp
import math

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for mouth & eyes (from MediaPipe FaceMesh)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

def euclidean_dist(pt1, pt2):
    return math.dist(pt1, pt2)

def smile_ratio(landmarks, w, h):
    left = (int(landmarks[LEFT_MOUTH].x * w), int(landmarks[LEFT_MOUTH].y * h))
    right = (int(landmarks[RIGHT_MOUTH].x * w), int(landmarks[RIGHT_MOUTH].y * h))
    top = (int(landmarks[UPPER_LIP].x * w), int(landmarks[UPPER_LIP].y * h))
    bottom = (int(landmarks[LOWER_LIP].x * w), int(landmarks[LOWER_LIP].y * h))

    mouth_width = euclidean_dist(left, right)
    mouth_open = euclidean_dist(top, bottom)
    return mouth_width / mouth_open if mouth_open != 0 else 0

def eye_aspect(landmarks, top_idx, bottom_idx, w, h):
    top = (int(landmarks[top_idx].x * w), int(landmarks[top_idx].y * h))
    bottom = (int(landmarks[bottom_idx].x * w), int(landmarks[bottom_idx].y * h))
    return euclidean_dist(top, bottom)

# Thresholds (tweak based on your webcam/face)
SMILE_THRESHOLD = 3.0
EYE_SQUINT_THRESHOLD = 8.0  # smaller = eyes squint more

smile_count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    label = ""

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            sr = smile_ratio(face_landmarks.landmark, w, h)

            if sr > SMILE_THRESHOLD:
                # Measure eyes
                left_eye = eye_aspect(face_landmarks.landmark, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
                right_eye = eye_aspect(face_landmarks.landmark, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)
                avg_eye = (left_eye + right_eye) / 2.0

                smile_count += 1
                if avg_eye < EYE_SQUINT_THRESHOLD:
                    label = "😊 Genuine Smile"
                else:
                    label = "🙂 Fake Smile"
            else:
                label = "😐 Neutral"

            # Draw face mesh (optional)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
            )

    # Show results
    cv2.putText(frame, f"Smiles: {smile_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, label, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
