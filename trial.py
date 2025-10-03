import cv2
from PIL import Image
from pytesseract import pytesseract

# Paths
path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
image_path = r"C:\Users\ASUS\Music\xbiz-trial\flask_ui\uploads\pooja_driv.jpg"

# Provide tesseract path
pytesseract.tesseract_cmd = path_to_tesseract

# Load image with OpenCV
img_cv = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

# Resize (improves small text detection)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# Noise removal
gray = cv2.medianBlur(gray, 3)

# Thresholding
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 31, 2
)

# OCR
text = pytesseract.image_to_string(thresh, lang="eng+hin")

print(text.strip())
