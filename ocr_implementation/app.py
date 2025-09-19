from flask import Flask, request, jsonify
import os
import cv2
import pytesseract
import easyocr
from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)

# Config
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize OCR engines once
easy_reader = easyocr.Reader(['en'], gpu=False)
paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')


def save_file(file):
    """Save uploaded file and return path"""
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath


@app.route("/ocr/easyocr", methods=["POST"])
def easyocr_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = save_file(file)

    results = easy_reader.readtext(filepath)
    text = "\n".join([res[1] for res in results])

    output_file = os.path.join(OUTPUT_FOLDER, "easyocr_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    return jsonify({"engine": "easyocr", "text": text})


@app.route("/ocr/paddleocr", methods=["POST"])
def paddleocr_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = save_file(file)

    results = paddle_reader.ocr(filepath, cls=True)
    lines = []
    for res in results:
        for line in res:
            lines.append(line[1][0])
    text = "\n".join(lines)

    output_file = os.path.join(OUTPUT_FOLDER, "paddleocr_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    return jsonify({"engine": "paddleocr", "text": text})


@app.route("/ocr/tesseract", methods=["POST"])
def tesseract_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filepath = save_file(file)

    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    output_file = os.path.join(OUTPUT_FOLDER, "tesseract_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    return jsonify({"engine": "tesseract", "text": text})


if __name__ == "__main__":
    app.run(debug=True)
