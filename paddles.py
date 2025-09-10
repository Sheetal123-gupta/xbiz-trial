from paddleocr import PaddleOCR
import os

# Config
IMAGE_PATH = "aadhar_s.png"   # replace with your image path
OUTPUT_DIR = "output-day1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "paddleocr_output.txt")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run PaddleOCR
results = ocr.ocr(IMAGE_PATH, cls=True)

# Extract text
lines = []
for res in results:
    for line in res:
        lines.append(line[1][0])  # line[1][0] contains recognized text

text = "\n".join(lines)

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(text)

print(f"PaddleOCR output saved to {OUTPUT_FILE}")
