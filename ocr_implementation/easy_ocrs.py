import easyocr
import os

# Config
IMAGE_PATH = "a2.jpg"   # replace with your image path
OUTPUT_DIR = "/ocr_implementation-1/output-day1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "/ocr_implementation/output-day1/easyocr_output.txt")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Run EasyOCR
results = reader.readtext(IMAGE_PATH)

# Extract text
text = "\n".join([res[1] for res in results])

# Save to file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(text)

print(f"EasyOCR output saved to {OUTPUT_FILE}")
