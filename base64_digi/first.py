#------------------------------according to task it is completed -------------------------

import json
import cv2
import numpy as np

# Load OCR JSON
with open("til_aadhar.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
paragraphs = []
words = []
lines = []

# Traverse OCR response
for page in data["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]:
    for block in page["blocks"]:
        for para in block["paragraphs"]:
            # Collect paragraph text
            para_text = ""
            for word in para["words"]:
                word_text = "".join([s["text"] for s in word["symbols"]])
                words.append({"text": word_text, "confidence": word["confidence"]})
                para_text += word_text + " "
            para_text = para_text.strip()
            paragraphs.append(para_text)

            # Add line with bounding box Y for sorting
            y_coord = para["boundingBox"]["vertices"][0]["y"]
            x_coord = para["boundingBox"]["vertices"][0]["x"]
            lines.append({
                "text": para_text,
                "y": y_coord,
                "x": x_coord,
                "bbox": para["boundingBox"]["vertices"]
            })

# Sort lines top → bottom
sorted_lines = sorted(lines, key=lambda x: (x["y"], x["x"]))

# Save JSON files
with open("paragraphs.json", "w", encoding="utf-8") as f:
    json.dump(paragraphs, f, ensure_ascii=False, indent=4)

with open("words.json", "w", encoding="utf-8") as f:
    json.dump(words, f, ensure_ascii=False, indent=4)

with open("text.json", "w", encoding="utf-8") as f:
    json.dump("\n".join(paragraphs), f, ensure_ascii=False, indent=4)

with open("sorted_text.json", "w", encoding="utf-8") as f:
    json.dump([line["text"] for line in sorted_lines], f, ensure_ascii=False, indent=4)

print("✅ JSON files created: paragraphs.json, words.json, text.json, sorted_text.json")

# Draw text & contours on image
image_path = "image.png"  # input image
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

for line in sorted_lines:
    # Get bounding box
    pts = [(v["x"], v["y"]) for v in line["bbox"]]
    pts = [(int(x), int(y)) for x, y in pts]

    # Draw contour
    cv2.polylines(img, [cv2.convexHull(np.array(pts))], True, (0, 255, 0), 2)

    # Put text above the box
    x, y = pts[0]
    cv2.putText(img, line["text"], (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

# Save final output
output_path = "output.jpg"
cv2.imwrite(output_path, img)
print(f"✅ Output saved as {output_path}")
