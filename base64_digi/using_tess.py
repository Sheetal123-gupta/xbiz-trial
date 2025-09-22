import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Optional: specify Tesseract path (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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

    # Invert if needed
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(binary == 255) < np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    # Use edge detection + Hough lines for angle estimation
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -60 < angle < 60:  # Filter out nearly vertical lines
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
    
    # Bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding for better contrast and illumination handling
    thresh = cv2.adaptiveThreshold(filtered, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    
    # Invert if background is white and text is black for Tesseract
    if np.sum(thresh == 255) > np.sum(thresh == 0):
        thresh = cv2.bitwise_not(thresh)
    
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return cleaned

# --- Preprocessing pipeline ---
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    
    # Store original for comparison
    original_img = img.copy()
    
    # Apply corrections
    img, rotation_applied = correct_rotation(img)
    img, skew_applied = correct_skew(img)
    
    # Save the corrected image if any corrections were applied
    corrections_applied = rotation_applied or skew_applied
    if corrections_applied:
        base_name = os.path.splitext(image_path)[0]
        corrected_path = base_name + "_corrected.jpg"
        cv2.imwrite(corrected_path, img)
        print(f"✅ Corrected image saved to: {corrected_path}")
        
        # Optional: Create a side-by-side comparison
        create_comparison_image(original_img, img, base_name + "_comparison.jpg")
    else:
        print("[INFO] No corrections needed - image was already properly oriented")
    
    return img

# --- Create side-by-side comparison ---
def create_comparison_image(original, corrected, output_path):
    h1, w1 = original.shape[:2]
    h2, w2 = corrected.shape[:2]
    
    target_height = min(h1, h2, 800)  # Cap at 800px
    
    scale1 = target_height / h1
    new_w1 = int(w1 * scale1)
    original_resized = cv2.resize(original, (new_w1, target_height))
    
    scale2 = target_height / h2
    new_w2 = int(w2 * scale2)
    corrected_resized = cv2.resize(corrected, (new_w2, target_height))
    
    total_width = new_w1 + new_w2 + 10  # 10px gap
    comparison = np.ones((target_height, total_width, 3), dtype=np.uint8) * 255
    
    comparison[:, :new_w1] = original_resized
    comparison[:, new_w1+10:] = corrected_resized
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "ORIGINAL", (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(comparison, "CORRECTED", (new_w1 + 20, 30), font, 1, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"📊 Comparison image saved to: {output_path}")

# --- OCR + box extraction with improved preprocessing ---
def extract_text_and_boxes(image):
    ocr_ready = prepare_image_for_ocr(image)
    
    data = pytesseract.image_to_data(
        ocr_ready,
        output_type=Output.DICT,
        lang="eng",
        config='--oem 3 --psm 6'
    )
    boxes = []
    text_content = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append({
                "text": word,
                "x_min": x,
                "y_min": y,
                "x_max": x + w,
                "y_max": y + h
            })
            text_content.append(word)
    full_text = " ".join(text_content)
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

# --- Draw contours and labels ---
def draw_boxes(image, boxes, output_path):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(img_rgb)
    
    sorted_boxes = sorted(boxes, key=lambda b: (b["y_min"], b["x_min"]))  # Top-down order
    for idx, b in enumerate(sorted_boxes, 1):
        # Draw rectangle
        rect = Rectangle(
            (b["x_min"], b["y_min"]),
            b["x_max"] - b["x_min"],
            b["y_max"] - b["y_min"],
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        # Add numbered label (e.g., 1. Name)
        label_text = f"{idx}. {b['text']}"
        ax.text(
            b["x_min"], b["y_min"] - 5,
            label_text,
            fontsize=6,
            color="yellow",
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
            verticalalignment="bottom"
        )
    
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()


# --- Main Pipeline ---
def process_document(image_path):
    print(f"🔄 Processing: {image_path}")
    image = preprocess_image(image_path)
    text, boxes = extract_text_and_boxes(image)
    doc_type = detect_document_type(text)
    annotated_path = os.path.splitext(image_path)[0] + "_annotated.png"
    draw_boxes(image, boxes, annotated_path)
    print(f"📄 Document Type: {doc_type}")
    print(f"🖼️ Annotated Image saved to: {annotated_path}")
    return {
        "document_type": doc_type,
        "text": text,
        "annotated_image": annotated_path
    }

# --- Run ---
if __name__ == "__main__":
    image_path = "images/tilted_aadhar.jpg"  # 🔁 Change to your test image
    result = process_document(image_path)

    print("\n--- Extracted Text ---")
    print(result["text"])
