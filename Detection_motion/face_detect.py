import cv2

# Open cammera
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale (faster for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facedetect.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(50, 50)   # optional: ignore very small detections
    )

    # Draw bounding boxes for all detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Face Detection", frame)

    # Quit if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
