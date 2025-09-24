import os
import base64
import json
import requests
from datetime import datetime
import shutil
import config
import cv2
import numpy as np
import logging

def encode_file_to_base64(filepath):
    """Convert image file to base64 string"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def save_image(base64_str, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # remove data:image/png;base64, if present
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(base64_str))

def log_error(message, log_folder):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, f"error_{datetime.now().strftime('%Y%m%d')}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {message}\n")

def create_txn_folder(txn_id, doc_name):
    """Create txn folder structure dynamically"""
    last4 = doc_name.split(".")[0].zfill(4)[-4:]  # ensure 4 digits
    folder_name = f"{txn_id}_{last4}"

    base_path = os.path.join(os.getcwd(), folder_name)
    subfolders = ["inputs", "assets", "outputs/text", "outputs/annotated", "errorlogs"]

    for sub in subfolders:
        os.makedirs(os.path.join(base_path, sub), exist_ok=True)

    return base_path

def safe_get_vertex_top_left(bbox_vertices):
    """Extract bounding box coordinates from vertices"""
    xs = [v.get("x", 0) for v in bbox_vertices if isinstance(v, dict)]
    ys = [v.get("y", 0) for v in bbox_vertices if isinstance(v, dict)]
    if not xs: xs = [0]
    if not ys: ys = [0]
    return min(xs), min(ys), max(xs), max(ys)

def draw_text_contours(base64_image_str, paragraphs, save_path):
    """Draw bounding boxes with text annotations on the image and save"""
    # Remove header if present
    if base64_image_str.startswith("data:image"):
        base64_image_str = base64_image_str.split(",", 1)[1]

    image_data = base64.b64decode(base64_image_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        print("Failed to decode image")
        return

    for para in paragraphs:
        x1, y1 = para["x_min"], para["y_min"]
        x2, y2 = para["x_max"], para["y_max"]
        text = para["text"]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put text (optional – truncate if too long)
        max_text_len = 30
        display_text = text[:max_text_len] + "..." if len(text) > max_text_len else text
        cv2.putText(image, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Save the annotated image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
def generate_ocr_sorted(paragraphs, output_path, y_threshold=15):
    """
    Group text into lines based on y-coordinate similarity.
    y_threshold controls how close in vertical position two items must be to be on the same line.
    """
    lines = []
    current_line = []
    last_y = None

    # Sort by y first, then x
    paragraphs = sorted(paragraphs, key=lambda p: (p["y_min"], p["x_min"]))

    for p in paragraphs:
        if last_y is None:
            current_line.append(p)
            last_y = p["y_min"]
            continue

        # Check if this paragraph is on the same line (close y)
        if abs(p["y_min"] - last_y) <= y_threshold:
            current_line.append(p)
        else:
            # Commit the finished line
            line_text = " ".join([c["text"] for c in sorted(current_line, key=lambda x: x["x_min"])])
            lines.append(line_text)
            # Start new line
            current_line = [p]
            last_y = p["y_min"]

    # Add the last line
    if current_line:
        line_text = " ".join([c["text"] for c in sorted(current_line, key=lambda x: x["x_min"])])
        lines.append(line_text)

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def extract_paragraphs_from_response(resp_json):
    """Extract paragraphs with their coordinates from OCR response"""
    out = []
    try:
        pages = resp_json["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]
    except Exception:
        pages = []
        if isinstance(resp_json.get("results"), list):
            for r in resp_json["results"]:
                d = r.get("Data", {})
                resp_list = d.get("responses", [])
                for rr in resp_list:
                    fta = rr.get("fullTextAnnotation")
                    if fta and isinstance(fta.get("pages"), list):
                        pages.extend(fta.get("pages"))

    for page_idx, page in enumerate(pages):
        page_width = page.get("width", None)
        page_height = page.get("height", None)
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                words = para.get("words", [])
                para_text_parts = []
                para_bbox = para.get("boundingBox", {}).get("vertices", [])
                word_xs, word_ys = [], []
                
                for w in words:
                    syms = w.get("symbols", [])
                    wtext = "".join([s.get("text", "") for s in syms])
                    if wtext:
                        para_text_parts.append(wtext)
                    wb = w.get("boundingBox", {}).get("vertices", [])
                    xs = [vv.get("x", 0) for vv in wb if isinstance(vv, dict)]
                    ys = [vv.get("y", 0) for vv in wb if isinstance(vv, dict)]
                    if xs: word_xs.extend(xs)
                    if ys: word_ys.extend(ys)

                para_text = " ".join(para_text_parts).strip()
                if not para_text:
                    continue
                    
                if para_bbox:
                    x_min, y_min, x_max, y_max = safe_get_vertex_top_left(para_bbox)
                else:
                    if word_xs and word_ys:
                        x_min, y_min, x_max, y_max = min(word_xs), min(word_ys), max(word_xs), max(word_ys)
                    else:
                        x_min, y_min, x_max, y_max = 0, 0, 0, 0

                out.append({
                    "page_idx": page_idx,
                    "text": para_text,
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                    "page_width": page_width,
                    "page_height": page_height
                })
    return out
def console_layout_output(sorted_paras, scale=0.08):
    """Generate console-style layout preserving spatial positioning"""
    canvas = {}
    for p in sorted_paras:
        row = int(p["y_min"] * scale)
        col = int(p["x_min"] * scale)
        canvas.setdefault(row, {})[col] = p["text"]
    lines = []
    for row in sorted(canvas.keys()):
        parts = []
        last_col = 0
        for col in sorted(canvas[row].keys()):
            spaces = max(0, (col - last_col))
            parts.append(" " * spaces + canvas[row][col])
            last_col = col + len(canvas[row][col]) // 5
        lines.append("".join(parts))
    return "\n".join(lines)
def format_extracted_text(response_data):
    """Format the response data into clean, readable text"""
    formatted_text = ""
    if "results" in response_data and response_data["results"]:
        result = response_data["results"][0]
        # If there's a 'text' field, use it directly
        if result.get("text"):
            formatted_text = result["text"]
        # If there are structured fields, format them nicely
        elif "extractedFields" in result or "fields" in result:
            fields = result.get("extractedFields", result.get("fields", {}))
            
            for key, value in fields.items():
                # Clean up the key name (remove underscores, capitalize)
                clean_key = key.replace("_", " ").title()
                formatted_text += f"{clean_key}: {value}\n"
        # If there are other text-related fields, include them
        else:
            # Look for any text-like fields in the response
            text_fields = []
            for key, value in result.items():
                if isinstance(value, str) and key.lower() in ['text', 'content', 'extractedtext', 'ocr_text']:
                    text_fields.append(f"{key.title()}: {value}")  
            if text_fields:
                formatted_text = "\n".join(text_fields)
            else:
                # Fallback to the original flatten_json if no specific text found
                formatted_text = flatten_json(response_data)
    
    # Clean up the text - remove extra whitespace and format nicely
    lines = [line.strip() for line in formatted_text.split('\n') if line.strip()]
    formatted_text = '\n'.join(lines)
    
    return formatted_text
def flatten_json(data, indent=0):
    """Recursively format JSON into readable text"""
    text = ""
    spacing = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in ['documentblob', 'annotatedimage']:
                # Skip large binary data
                text += f"{spacing}{key}: [Binary data - omitted]\n"
                continue            
            text += f"{spacing}{key}:\n"
            text += flatten_json(value, indent + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            text += f"{spacing}- Item {i+1}:\n"
            text += flatten_json(item, indent + 1)
    else:
        text += f"{spacing}{data}\n"
    return text
def save_formatted_response_as_text(response_data, output_path):
    """Save the response as clean, formatted text"""
    formatted_text = format_extracted_text(response_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

def save_full_response_as_text(response_data, output_path):
    """Save the complete response as structured text (for debugging)"""
    readable_text = flatten_json(response_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readable_text)

def get_file_logger(filename, base_path):
    """Return a separate logger for each file"""
    log_folder = os.path.join(base_path, "errorlogs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, f"log_{os.path.splitext(filename)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def process_document(filename, txn_id):
    """Main processing pipeline"""
    base_path = create_txn_folder(txn_id, filename)
    logger = get_file_logger(filename, base_path)
    logger.info(f"started with processing {filename} (txn:{txn_id})")

    # Copy input file to txn folder
    src_path = os.path.join(config.INPUT_FOLDER, filename)
    dst_path = os.path.join(base_path, "inputs", filename)
    shutil.copy(src_path, dst_path)
    logging.info(f"copied file into the {dst_path}")
    # Encode the copied file
    doc_blob = encode_file_to_base64(dst_path)
    request_payload = {
        "txnId": txn_id,
        "docType": os.path.splitext(filename)[1],
        "source": "OCR_RAW",
        "documentName": filename,
        "caseNo": "case001",
        "documentBlob": doc_blob
    }
    save_json(request_payload, os.path.join(base_path, "assets", "request.json"))
    logging.info(f"now saved into urs {filename}")
    try:
        response = requests.post(config.API_URL, json=request_payload, timeout=60)
        response_data = response.json()
        save_json(response_data, os.path.join(base_path, "assets", "response.json"))
        logging.info(f"saved into the {filename}")
        if "results" in response_data and response_data["results"]:
            result = response_data["results"][0]            
            # Extract paragraphs with coordinates and save layout version
            try:
                paras = extract_paragraphs_from_response(response_data)
                if paras:
                    # Sort paragraphs by position (top-to-bottom, left-to-right)
                    sorted_paras = sorted(paras, key=lambda p: (p["y_min"], p["x_min"]))    
                    # Generate layout-preserved text
                    layout_text = console_layout_output(sorted_paras)
                    # Save layout text
                    layout_path = os.path.join(base_path, "outputs", "text", f"{os.path.splitext(filename)[0]}_layout.txt")
                    save_text(layout_text, layout_path) 
                    logging.info(f"layout file saved into {filename}") 

                    #sorted_ocr
                    ocr_sorted_path = os.path.join(base_path, "outputs", "text", f"{os.path.splitext(filename)[0]}_ocr_sorted.txt")
                    generate_ocr_sorted(sorted_paras, ocr_sorted_path)
                    logging.info(f"OCR sorted file saved into {ocr_sorted_path}")
                    contour_path=os.path.join(base_path,"outputs","annotated",f"contoured_{filename}")
                    # Read the input image as base64
                    with open(dst_path, "rb") as f:
                        base64_img = base64.b64encode(f.read()).decode("utf-8")

                    draw_text_contours(base64_img, sorted_paras, contour_path)
                    logging.info(f"contour ya bounding box image are saved into {filename}")

                    if result.get("annotatedImage"):
                      contour_path = os.path.join(base_path, "outputs", "annotated", f"contoured_{filename}")
                      draw_text_contours(result["annotatedImage"], sorted_paras, contour_path)

                else:
                    logging.warning(f"kuch paragraph nahi mila isme{filename}")  
            except Exception as e:
                logging.error(f"failed for {filename} : {str(e)}")
            # Save annotated image if present
            if result.get("annotatedImage"):
                image_path = os.path.join(base_path, "outputs", "annotated", f"annotated_{filename}")
                save_image(result["annotatedImage"], image_path)
                logging.info(f"completed")
            
        else:
            log_error(f"Failed for {filename}: {response_data.get('errorMessage', 'Unknown error')}",
                      os.path.join(base_path, "errorlogs"))
            print(f" Error for {filename}: see logs")
    except Exception as e:
        log_error(f"Exception for {filename}: {str(e)}", os.path.join(base_path, "errorlogs"))
        logging.info(f"check logs")

if __name__ == "__main__":
    txn_id = "TXNID1"
    for file in os.listdir(config.INPUT_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
            process_document(file, txn_id)