import cv2
import pytesseract
#reading of image
img=cv2.imread("pan5.jpg")

#convert into grascale
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray)
print(f"---------------------------------------------")
gray=cv2.bilateralFilter(gray,9,75,75)
print(gray)
print("=============================")
gray=cv2.GaussianBlur(gray,(5,5),0)
print(gray)
thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15)
print(thresh)

edges=cv2.Canny(thresh,50,150)
print(edges)

contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(contours)

for i in contours:
  x,y,w,h=cv2.boundingRect(i)
  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
print(i)

text=pytesseract.image_to_string(thresh)
print(text)

