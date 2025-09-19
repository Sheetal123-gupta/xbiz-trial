import pytesseract
import cv2
from PIL import Image

# Test if tesseract is installed
try:
    print("Tesseract version:", pytesseract.get_tesseract_version())
except:
    print("Tesseract not found! Please install it.")

# Test with your image
img = cv2.imread('check.jpg')
if img is None:
    print("Cannot load image 'pan5.jpg' - check the file path!")
else:
    print("Image loaded successfully")
    # Simple OCR test
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    print("Simple OCR result:")
    print(text)