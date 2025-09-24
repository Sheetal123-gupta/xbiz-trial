'''
#Reads OCR JSON output.
#Extracting Text Paragraphs.
#Storing Text with Coordinates
#Sorts them top-to-bottom, left-to-right.(4. Sorting by Page Layout)
#Places them on a "canvas" with spacing.(5. Creating a Canvas Grid)
#Aligning Output into Readable Lines
#7. Saving & Printing


import json
with open("ocr_response.json", "r", encoding="utf-8") as f:
    data = json.load(f)
lines = []
for page_idx, page in enumerate(data["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]):
    for block_idx, block in enumerate(page["blocks"]):
        for para_idx, para in enumerate(block["paragraphs"]):
            para_text = ""
            for word in para["words"]:
                word_text = "".join([s["text"] for s in word["symbols"]])
                para_text += word_text + " "
            para_text = para_text.strip()
            if para_text:
                y_coord = para["boundingBox"]["vertices"][0].get("y", 0)
                x_coord = para["boundingBox"]["vertices"][0].get("x", 0)
                lines.append({
                    "text": para_text,
                    "y": y_coord,
                    "x": x_coord
                })
sorted_lines = sorted(lines, key=lambda x: (x["y"], x["x"]))
max_y = max(line["y"] for line in sorted_lines) + 50
scale = 0.1  
canvas = {}
for line in sorted_lines:
    row = int(line["y"] * scale)  
    col = int(line["x"] * scale)
    if row not in canvas:
        canvas[row] = {}
    canvas[row][col] = line["text"]
aligned_output = []
for row in sorted(canvas.keys()):
    line_parts = []
    last_col = 0
    for col in sorted(canvas[row].keys()):
        spaces = " " * ((col - last_col) // 10)
        line_parts.append(spaces + canvas[row][col])
        last_col = col
    aligned_output.append("".join(line_parts))
with open("aligned_layout.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(aligned_output))
print("✅ Layout-style text saved to aligned_layout.txt\n")
print("\n".join(aligned_output))
'''
import cv2
import base64
import json
import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --------- USER CONFIG ----------
API_URL = "https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction"
IMAGE_PATH = "aadhar_dhapu.png"         # <-- change to your file (jpg/png)
DOCUMENT_NAME = "aadhar_dhapu.png"
TXN_ID = "TXN0000001"
TIMEOUT = 60
OUTPUT_TEXT = "aligned_layout2.txt"
OUTPUT_IMAGE = "annotated_layout2.png"
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

#OCR JSON me se pages nikalta hai.Har page me blocks -> paragraphs -> words -> symbols tak traverse karta hai.Har paragraph ka text join karta hai.Har paragraph ka position (x_min, y_min, x_max, y_max) aur page size store karta hai.Saari paragraphs ka list return karta hai

def extract_paragraphs_from_response(resp_json):
    out = []
    try:
        pages = resp_json["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]
    except Exception:
        # best-effort: try to locate pages anywhere
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
                # compile paragraph text from words/symbols
                words = para.get("words", [])
                para_text_parts = []
                # sometimes paragraph bounding box is present:
                para_bbox = para.get("boundingBox", {}).get("vertices", [])
                # fallback bounding box built from words if needed:
                word_xs, word_ys = [], []
                for w in words:
                    # symbol-level text join
                    syms = w.get("symbols", [])
                    wtext = "".join([s.get("text", "") for s in syms])
                    if wtext:
                        para_text_parts.append(wtext)
                    # collect word bbox for fallback
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
# paragraph ka osition(x,y) or text lena he ek dictionary me text ko uske row or column ke hisab se place krna he, usko line by line text me lo with space add karjke then txt file me save karo
def console_layout_output(sorted_paras, out_path=OUTPUT_TEXT, scale=CONSOLE_SCALE):
    # Build a sparse 2D canvas keyed by scaled row (y) and col (x)
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
            # compute number of spaces; integer division to avoid long strings
            spaces = max(0, (col - last_col))
            parts.append(" " * spaces + canvas[row][col])
            last_col = col + len(canvas[row][col]) // 5  # rough advance
        lines.append("".join(parts))
    txt = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"✅ Saved console-style layout -> {out_path}")
    return txt

#Original image load karta hai.Extracted paragraphs ke positions pe green rectangles draw karta hai.Rectangle ke upar yellow text likhta hai.Final image ko save karta hai aur console me success message print karta hai.

def plot_contours_on_image(image_path, sorted_paras, output_image="contour_img.png"):
    # Load original image (keeps PAN card visible)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=200)
    ax.imshow(img_rgb)

    for p in sorted_paras:
        x = p["x_min"]
        y = p["y_min"]
        box_w = p["x_max"] - p["x_min"]
        box_h = p["y_max"] - p["y_min"]
        # Green contour
        rect = Rectangle((x, y), box_w, box_h,
                         linewidth=1.5, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        # Text above contour
        ax.text(x, y-5, p["text"], fontsize=6, color="yellow",
                bbox=dict(facecolor="black", alpha=0.4, pad=1),
                verticalalignment="bottom")

    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"✅ Contours with text drawn on original image -> {output_image}")

#Blank white canvas banata hai.Har paragraph ke liye green box draw karta hai.Box ke top-left me red block ID lagata hai.Box ke andar blue text likhta hai.Final image save kar ke output deta hai

def plot_layout(sorted_paras, image_size_hint=None, output_image=OUTPUT_IMAGE):
    # Determine canvas size (prefer page_width/page_height from data)
    page_w = None
    page_h = None
    for p in sorted_paras:
        if p.get("page_width"):
            page_w = p["page_width"]
            page_h = p["page_height"]
            break

    # If we don't have page size from JSON, use image file
    if not page_w and image_size_hint:
        page_w, page_h = image_size_hint

    # final fallback
    if not page_w:
        page_w, page_h = 1200, 1600

    # create white background image
    canvas_img = np.ones((page_h, page_w, 3), dtype=np.uint8) * 255

    fig_w = max(6, page_w / 200)   # scale for figure size
    fig_h = max(8, page_h / 200)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.imshow(canvas_img, origin="upper")

    # 🔹 assign block numbers
    for idx, p in enumerate(sorted_paras, start=1):
        x = p["x_min"]
        y = p["y_min"]
        w = max(1, p["x_max"] - p["x_min"])
        h = max(1, p["y_max"] - p["y_min"])

        # draw bounding box (green)
        rect = Rectangle((x, y), w, h, linewidth=1.2, edgecolor="green", facecolor="none")
        ax.add_patch(rect)

        # draw block ID (red, bold) at top-left corner
        ax.text(x, y - 5, str(idx), fontsize=9, color="red",
                fontweight="bold", verticalalignment="bottom")

        # 🔹 draw the text itself inside the bounding box at its natural position
        ax.text(x + 2, y + h/2, p["text"], fontsize=7, color="blue",
                verticalalignment="center", wrap=True)

    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)  # invert Y to match image coords (origin top-left)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_image, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"✅ Saved plotted layout with block IDs + text at position -> {output_image}")

def main(image_path=IMAGE_PATH):
    # 1) create base64
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    b64 = image_to_base64(image_path)

    # 2) call API
    print("Calling OCR API...")
    resp_json = call_api_with_base64(b64, documentName=os.path.basename(image_path))
    print("Received response.")

    # 3) extract paras
    paras = extract_paragraphs_from_response(resp_json)
    if not paras:
        print("No paragraphs found in API response.")
        return

    # 4) sort by y (top->bottom) then x
    sorted_paras = sorted(paras, key=lambda p: (p["y_min"], p["x_min"]))

    # 5) Console & file output (text + coordinates)
    print("\n---- Extracted lines with coordinates (top-left) ----\n")
    for p in sorted_paras:
        print(f"(x={p['x_min']}, y={p['y_min']})  -> {p['text']}")

    # Save console-simulated layout(koi comment remove karne ki jarurat nahi)
    #console_text = console_layout_output(sorted_paras, out_path=OUTPUT_TEXT, scale=CONSOLE_SCALE)
    #print("\n--- Console-simulated layout (saved) ---\n")
    #print(console_text)

    # 6) Plot layout image (use actual image size as hint)
    with Image.open(image_path) as im:
        image_size_hint = im.size  # (width, height)
    plot_layout(sorted_paras, image_size_hint=image_size_hint, output_image=OUTPUT_IMAGE)

    #mene jo contour  par text dala 
    plot_contours_on_image("images/aadhar_dhapu.png", sorted_paras, "contour_img2.png")

if __name__ == "__main__":
    main()
