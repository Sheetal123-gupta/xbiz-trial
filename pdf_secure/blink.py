import cv2
import os

# Create output folder if it doesn't exist
output_folder = "blink_output"
os.makedirs(output_folder, exist_ok=True)

# Load Haar cascade for eyes
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

frame_count = 0  # for naming saved images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    if len(eyes) == 0:
        cv2.putText(frame, "Blink", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

        # Save frame as image
        img_name = os.path.join(output_folder, f"blink_{frame_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        frame_count += 1

    # Show frame
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
