


import cv2
import mediapipe as mp
import time


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing = mp.solutions.drawing_utils

# Eye landmark indices (left and right)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def euclidean(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

def get_ear(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    vertical = (euclidean(p[1], p[5]) + euclidean(p[2], p[4])) / 2
    horizontal = euclidean(p[0], p[3])
    return vertical / horizontal
cap = cv2.VideoCapture(0)
blink_count = 0
blink_flag = False
ear_threshold = 0.25
cooldown = 0.5  # seconds
last_blink_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = get_ear(face_landmarks.landmark, LEFT_EYE)
            right_ear = get_ear(face_landmarks.landmark, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < ear_threshold and not blink_flag:
                if time.time() - last_blink_time > cooldown:
                    blink_count += 1
                    print(f"Blink #{blink_count}")
                    blink_flag = True
                    last_blink_time = time.time()
            elif avg_ear >= ear_threshold:
                blink_flag = False

            drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

    cv2.putText(frame, f"Blinks: {blink_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
