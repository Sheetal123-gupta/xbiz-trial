from ultralytics import YOLO
import cv2


# Load YOLO face model (download yolov8n-face.pt beforehand)
model = YOLO("C:\\Users\\ASUS\\Music\\xbiz-trial\\object_detection\\yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame=cv2.flip(frame,1)

    # Run YOLO detection
    results = model(frame, conf=0.5)

    
    # Annotate detections on frame
    annotated = results[0].plot()
    annotated=cv2.resize(annotated,(1280,720))


    # Show results
    cv2.imshow("YOLO Face Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
