from rapidfuzz import fuzz
import cv2
import base64
import json
import requests
import os
import re
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------- USER CONFIG ----------
API_URL = "https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction"
IMAGE_PATH = "images/voter.png"
DOCUMENT_NAME = "images/voter.png"
TXN_ID = "TXN0000001"
TIMEOUT = 60
OUTPUT_TEXT = "aligned_layout5.txt"
OUTPUT_IMAGE = "annotated_layout5.png"
CONSOLE_SCALE = 0.08


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_api_with_base64(b64str, documentName=DOCUMENT_NAME, txnId=TXN_ID):
    payload = {
        "txnId": txnId,
        "docType": os.path.splitext(documentName)[1] or ".JPG",
        "source": "OCR_RAW",
        "documentName": documentName,
        "caseNo": "case001",
        "documentBlob": b64str
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def safe_get_vertex_top_left(bbox_vertices):
    xs = [v.get("x", 0) for v in bbox_vertices if isinstance(v, dict)]
    ys = [v.get("y", 0) for v in bbox_vertices if isinstance(v, dict)]
    if not xs: xs = [0]
    if not ys: ys = [0]
    return min(xs), min(ys), max(xs), max(ys)

def extract_paragraphs_from_response(resp_json):
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

def detect_document_type_from_text(text):
    text_lower = text.lower()
    docs = {
        "PAN Card": ["income tax department", "permanent account number","आयकर विभाग",
                     "भारत सरकार","सत्यमेव जयते"],
        "Aadhaar Card": [
            "unique identification authority of india", "aadhaar", "आधार", "जन्म", "जन्म वर्ष", "year of birth",
            "महिला", "पुरुष", "female", "male", "सत्यमेव जयते", "father", "आम आदमी का अधिकार"
        ],
        "Voter ID": ["election commission", "elector", "voter id", "constituency", "identity card","भारत निर्वाचन आयोग", "elector's name", "photo identity", "epic", "voter", "pehchan patra", "पहचान पत्र"],

        "Bank Document": ["bank", "account", "ifsc", "branch"],
        "Driving License": ["driving licence", "dl", "date of issue", "driving license"]
    }

    scores = {}
    for doc_type, keywords in docs.items():
        score = 0
        for kw in keywords:
            ratio = fuzz.partial_ratio(kw, text_lower)
            if ratio > 55:  # Lowered threshold
                score += 1
        scores[doc_type] = score

    # Aadhaar fallback via 12-digit number detection
    if re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text):
        return "Aadhaar Card"
    if re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text):
        return "PAN Card"
    #if re.search(r"\b[]")

    detected_type = max(scores, key=scores.get)
    if scores[detected_type] < 2:
        return "Unknown"
    return detected_type

def console_layout_output(sorted_paras, out_path=OUTPUT_TEXT, scale=CONSOLE_SCALE):
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
    txt = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"✅ Saved console-style layout -> {out_path}")
    return txt


def plot_contours_on_image(image_path, sorted_paras, output_image="contour_img.png"):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=200)
    ax.imshow(img_rgb)

    for p in sorted_paras:
        x = p["x_min"]
        y = p["y_min"]
        box_w = p["x_max"] - p["x_min"]
        box_h = p["y_max"] - p["y_min"]
        rect = Rectangle((x, y), box_w, box_h, linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 5, p["text"], fontsize=6, color="yellow",
                bbox=dict(facecolor="black", alpha=0.4, pad=1), verticalalignment="bottom")

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"✅ Contours with text drawn on original image -> {output_image}")

def plot_layout(sorted_paras, image_size_hint=None, output_image=OUTPUT_IMAGE):
    page_w, page_h = None, None
    for p in sorted_paras:
        if p.get("page_width"):
            page_w = p["page_width"]
            page_h = p["page_height"]
            break

    if not page_w and image_size_hint:
        page_w, page_h = image_size_hint

    if not page_w:
        page_w, page_h = 1200, 1600

    canvas_img = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255
    fig_w = max(6, page_w / 200)
    fig_h = max(8, page_h / 200)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.imshow(canvas_img, origin="upper")

    for idx, p in enumerate(sorted_paras, start=1):
        x = p["x_min"]
        y = p["y_min"]
        w = max(1, p["x_max"] - p["x_min"])
        h = max(1, p["y_max"] - p["y_min"])

        rect = Rectangle((x, y), w, h, linewidth=1.2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 5, str(idx), fontsize=9, color="red",
                fontweight="bold", verticalalignment="bottom")
        ax.text(x + 2, y + h / 2, p["text"], fontsize=7, color="blue",
                verticalalignment="center", wrap=True)

    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved plotted layout with block IDs + text at position -> {output_image}")

def main(image_path=IMAGE_PATH):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    b64 = image_to_base64(image_path)

    print("Calling OCR API...")
    resp_json = call_api_with_base64(b64, documentName=os.path.basename(image_path))
    print("Received response.")

    paras = extract_paragraphs_from_response(resp_json)
    if not paras:
        print("No paragraphs found in API response.")
        return

    sorted_paras = sorted(paras, key=lambda p: (p["y_min"], p["x_min"]))
    # Combine all texts for detection
    full_text = "\n".join([p["text"] for p in sorted_paras])
    
    #Optional debug
    print("\n🔍 OCR Extracted Text:\n", full_text)

    #Detect document type
    doc_type = detect_document_type_from_text(full_text)
    print(f"\n📄 Detected Document Type: {doc_type}")

    #Print lines with coordinates
    print("\n---- Extracted lines with coordinates (top-left) ----\n")
    for p in sorted_paras:
        print(f"(x={p['x_min']}, y={p['y_min']})  -> {p['text']}")

    #Plot layout
    with Image.open(image_path) as im:
        image_size_hint = im.size
    plot_layout(sorted_paras, image_size_hint=image_size_hint, output_image=OUTPUT_IMAGE)
    plot_contours_on_image(image_path, sorted_paras, "contour_img2.png")
if __name__ == "__main__":
    main()
