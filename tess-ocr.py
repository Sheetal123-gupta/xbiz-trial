import cv2
import pytesseract
import os

# Config
IMAGE_PATH = "aadhar_s.png"   # replace with your image path
OUTPUT_DIR = "output-day1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tesseract_output.txt")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read image
img = cv2.imread(IMAGE_PATH)

# Convert to grayscale for better accuracy
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run Tesseract OCR
text = pytesseract.image_to_string(gray)

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Tesseract OCR output saved to {OUTPUT_FILE}")
