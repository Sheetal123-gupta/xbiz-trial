import cv2
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

save_path = 'saved_faces'
os.makedirs(save_path, exist_ok=True)

i = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    face_imgs = []
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        if crop_img.size > 0:
            resized_img = cv2.resize(crop_img, (150, 150))
            face_imgs.append(resized_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)

    if face_imgs:
        # Put all detected faces in one row (side by side)
        combined = np.hstack(face_imgs)

        # Save combined image once per frame
        filename = os.path.join(save_path, f'faces_{i}.jpg')
        cv2.imwrite(filename, combined)
        print(f"Saved: {filename}")
        i += 1

        # Show combined faces in separate window (optional)
        cv2.imshow("Detected Faces", combined)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
