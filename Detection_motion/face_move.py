'''
#download the pretrained data model pf opencv dnn (download from github )and placed them inside the models/ 
#initialize face detection 
#capture the video and face detection
#then access the movement of it 

import cv2
import os
prototxt="C:\\Users\\ASUS\\Music\\xbiz-trial\\pdf_secure\\models\\deploy.prototxt"
caffemodel="C:\\Users\\ASUS\\Music\\xbiz-trial\\pdf_secure\\models\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
net=cv2.dnn.readNetFromCaffe(prototxt,caffemodel)

ans='saved_img'
os.makedirs(ans,exist_ok=True)
cap=cv2.VideoCapture(0)
prev_val=None

while True:
  ret,frame=cap.read()
  h,w=frame.shape[:2]
#blob = cv2.dnn.blobFromImage(frame(input image from webcam), 1.0(scaling factor), (300, 300)(input img ka width*height),(104.0, 177.0, 123.0)(mean subtraction value pata nahi kyu lete he ? ), False(by default opencv rgb ke jagah bgr leta he isliye), False(whetjer to crop img or not))
  blob=cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0,177.0,123.0),False,False)
  net.setInput(blob)
  detections=net.forward()
  for i in range(detections.shape[2]):
    confidence=detections[0,0,i,2]
    if confidence>0.3:
      box=detections[0,0,i,3:7]*[w,h,w,h]
      x1,y1,x2,y2=box.astype("int")
      cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

      #movement_detection
      if prev_val:
        dx=x1-prev_val[0]
        dy=y1-prev_val[1]
        if abs(dx)>10 or abs(dy)>10:
          cv2.putText(frame,"moved !! ",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
          filename=os.path.join(ans,f"moved_{cv2.getTickCount()}.png")
          cv2.imwrite(filename,frame)
      prev_val=x1,y1
  cv2.imshow("detected faces ",frame)
  if cv2.waitKey(1)==27:
    break
cap.release()
cv2.destroyAllWindows()
'''


import cv2
import os

prototxt = r"C:\Users\ASUS\Music\xbiz-trial\pdf_secure\models\deploy.prototxt"
caffemodel = r"C:\Users\ASUS\Music\xbiz-trial\pdf_secure\models\res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

ans = 'saved_img'
os.makedirs(ans, exist_ok=True)

cap = cv2.VideoCapture(0)
prev_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Compute face center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                direction = None

                if abs(dx) > abs(dy):  # Horizontal movement
                    if dx > 14:
                        direction = "RIGHT"
                    elif dx < -14:
                        direction = "LEFT"
                else:  # Vertical movement
                    if dy > 14:
                        direction = "DOWN"
                    elif dy < -14:
                        direction = "UP"

                if direction:
                    cv2.putText(frame, f"Moved {direction}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    filename = os.path.join(ans, f"{direction}_{cv2.getTickCount()}.png")
                    cv2.imwrite(filename, frame)

            prev_center = (cx, cy)

    cv2.imshow("Detected Faces", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
