#supporting with rotation
from flask import Flask, request, render_template, redirect, url_for
import os
import uuid
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import getpass

# Setup Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --------- [ KEEP YOUR FULL OCR FUNCTIONS HERE ] ---------
# Copy all your existing functions here:
# - correct_rotation()
# - correct_skew()
# - prepare_image_for_ocr()
# - preprocess_image()
# - extract_text_and_boxes()
# - detect_document_type()
# - create_comparison_image()
# - draw_boxes()
# --- Document Type Keywords ---
DOC_KEYWORDS = {
    "PAN Card": ["income tax department", "permanent account number", "आयकर विभाग", "भारत सरकार"],
    "Aadhaar Card": ["aadhaar", "आधार", "unique identification", "year of birth", "जन्म", "सत्यमेव जयते"],
    "Voter ID": ["election commission", "voter", "epic", "पहचान पत्र", "identity card"],
    "Bank Document": ["bank", "ifsc", "branch", "account"],
    "Driving License": ["driving licence", "transport department", "license number"]
}

FUZZY_THRESHOLD = 55
MIN_KEYWORD_MATCHES = 2

# --- Detect and correct major rotation (90, 180, 270) ---
def correct_rotation(image):
    rotation_applied = False
    try:
        osd = pytesseract.image_to_osd(image)
        angle = int(re.search(r"Rotate: (\d+)", osd).group(1))
        if angle != 0:
            print(f"Correcting rotation: {angle} degrees")
            rotation_applied = True
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"[WARN] Rotation detection failed: {e}")
    return image, rotation_applied

# --- Deskew (small tilt) ---
def correct_skew(image):
    skew_applied = False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -60 < angle < 60:
                angles.append(angle)

        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 0.5:
                print(f"[INFO] Rotating to fix arbitrary skew: {median_angle:.2f} degrees")
                skew_applied = True
                (h, w) = image.shape[:2]
                M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image, skew_applied

# --- Additional preprocessing to enhance OCR accuracy ---
def prepare_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned

# --- Preprocessing pipeline ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")

    original_img = img.copy()
    img, rotation_applied = correct_rotation(img)
    img, skew_applied = correct_skew(img)

    corrections_applied = rotation_applied or skew_applied
    base_name = os.path.splitext(image_path)[0]
    comparison_path = None

    if corrections_applied:
        corrected_path = base_name + "_corrected.jpg"
        cv2.imwrite(corrected_path, img)
        print(f"✅ Corrected image saved to: {corrected_path}")
        comparison_path = base_name + "_comparison.jpg"
        create_comparison_image(original_img, img, comparison_path)
    else:
        print("[INFO] No corrections needed - image was already properly oriented")
    return img, comparison_path


# --- Create side-by-side comparison ---
def create_comparison_image(original, corrected, output_path):
    h1, w1 = original.shape[:2]
    h2, w2 = corrected.shape[:2]
    target_height = min(h1, h2, 800)

    scale1 = target_height / h1
    new_w1 = int(w1 * scale1)
    original_resized = cv2.resize(original, (new_w1, target_height))

    scale2 = target_height / h2
    new_w2 = int(w2 * scale2)
    corrected_resized = cv2.resize(corrected, (new_w2, target_height))

    total_width = new_w1 + new_w2 + 10
    comparison = np.ones((target_height, total_width, 3), dtype=np.uint8) * 255

    comparison[:, :new_w1] = original_resized
    comparison[:, new_w1 + 10:] = corrected_resized

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "ORIGINAL", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "CORRECTED", (new_w1 + 20, 30), font, 1, (0, 255, 0), 2)

    cv2.imwrite(output_path, comparison)
    print(f"📊 Comparison image saved to: {output_path}")

# --- OCR with line-by-line group ---
def extract_text_and_boxes(image):
    ocr_ready = prepare_image_for_ocr(image)

    data = pytesseract.image_to_data(
        ocr_ready,
        output_type=Output.DICT,
        lang="eng",
        config='--oem 3 --psm 6'
    )

    boxes = []
    lines = {}

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if not word or not word.isascii():
            continue

        block = data["block_num"][i]
        par = data["par_num"][i]
        line = data["line_num"][i]
        key = (block, par, line)

        if key not in lines:
            lines[key] = []
        lines[key].append(word)

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        boxes.append({
            "text": word,
            "x_min": x,
            "y_min": y,
            "x_max": x + w,
            "y_max": y + h
        })

    full_text_lines = [' '.join(lines[key]) for key in sorted(lines.keys())]
    full_text = "\n".join(full_text_lines)

    return full_text, boxes

# --- Document Type Detection ---
def detect_document_type(text):
    text = text.lower()
    scores = {}
    for doc_type, keywords in DOC_KEYWORDS.items():
        score = sum(1 for kw in keywords if fuzz.partial_ratio(kw.lower(), text) >= FUZZY_THRESHOLD)
        scores[doc_type] = score
    if re.search(r"\b\d{4} \d{4} \d{4}\b", text):
        return "Aadhaar Card"
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text):
        return "PAN Card"
    if re.search(r"\b[A-Z]{2}/\d+/\d+/\d+\b", text):
        return "Voter ID"
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] >= MIN_KEYWORD_MATCHES else "Unknown"

# --- Draw contours only ---
def draw_contours_only(image, output_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_path, contour_image)
    print(f"📌 Contour-only image saved to: {output_path}")

# --- Draw boxes ---
def draw_boxes(image, boxes, output_path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img_rgb)

    for idx, b in enumerate(boxes, start=1):
        x_min, y_min = b["x_min"], b["y_min"]
        x_max, y_max = b["x_max"], b["y_max"]
        box_w = x_max - x_min
        box_h = y_max - y_min

        # Green rectangle around the text
        rect = Rectangle((x_min, y_min), box_w, box_h,
                         linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

        # Draw text label (yellow) above the box
        ax.text(x_min, y_min - 5, f"{idx}. {b['text']}", fontsize=6, color="yellow",
                bbox=dict(facecolor="black", alpha=0.4, pad=1), verticalalignment="bottom")

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()
    print(f"✅ Annotated image saved to: {output_path}")
# --- Add the wrapper to process uploaded file ---
def process_document(image_path):
    image, comparison_path = preprocess_image(image_path)
    text, boxes = extract_text_and_boxes(image)
    doc_type = detect_document_type(text)

    base = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = f"{base}_annotated.png"
    annotated_full = os.path.join(UPLOAD_FOLDER, annotated_path)
    draw_boxes(image, boxes, annotated_full)

    # --- Save extracted text to .txt file ---
    text_filename = f"{base}_extracted.txt"
    text_fullpath = os.path.join(UPLOAD_FOLDER, text_filename)
    with open(text_fullpath, "w", encoding="utf-8") as f:
        f.write(f"Document Type: {doc_type}\n\n")
        f.write(text)

    print(f"📝 Extracted text saved to: {text_fullpath}")


    return {
        "document_type": doc_type,
        "text": text,
        "annotated_image": annotated_path,
        "comparison_image": os.path.basename(comparison_path) if comparison_path else None,
        "text_file":text_filename
    }

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            result = process_document(filepath)
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",port=5000)  

    #asset- image jo user base64 convertion
    #request- 1.request jo request hit kar raha he 2.response jo output hoiga
    #output- text file(sirf ocr) jo output basically apna text and annotated images print ke jjagah logger save ito text file 
    #thinkng logging ? 
    