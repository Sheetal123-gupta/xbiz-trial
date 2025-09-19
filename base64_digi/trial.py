import json
import cv2
import numpy as np
import base64

json_path = "response.json"  # Replace with your JSON file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

base64_str = data.get("documentBlob", "")
if not base64_str:
    raise ValueError("No base64 image found in JSON")

# Remove any data prefix if present
if "," in base64_str:
    base64_str = base64_str.split(",")[1]

# Decode to bytes
img_data = base64.b64decode(base64_str)
nparr = np.frombuffer(img_data, np.uint8)
if nparr.size == 0:
    raise ValueError("Decoded byte array is empty")

# Convert to image
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
if img is None:
    raise ValueError("Failed to decode image from base64 bytes")

print(f"✅ Image loaded successfully: {json_path}")
print(f"Image dimensions: {img.shape}")

paragraphs = []
words = []
lines = []

# Traverse pages, blocks, paragraphs
for page in data["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]:
    for block in page["blocks"]:
        for para in block["paragraphs"]:
            para_text = ""
            for word in para["words"]:
                word_text = "".join([s["text"] for s in word["symbols"]])
                
                # Word bounding box
                bbox = word["boundingBox"]["vertices"]
                bbox_coords = [{"x": v.get("x", 0), "y": v.get("y", 0)} for v in bbox]
                
                words.append({
                    "text": word_text,
                    "confidence": word.get("confidence", 0),
                    "bounding_box": bbox_coords
                })
                para_text += word_text + " "
            
            para_text = para_text.strip()
            paragraphs.append(para_text)

            # Add line info for sorting
            y_coord = para["boundingBox"]["vertices"][0].get("y", 0)
            x_coord = para["boundingBox"]["vertices"][0].get("x", 0)
            lines.append({
                "text": para_text,
                "y": y_coord,
                "x": x_coord,
                "bbox": para["boundingBox"]["vertices"]
            })

# Sort lines top → bottom, left → right
sorted_lines = sorted(lines, key=lambda x: (x["y"], x["x"]))

for line in sorted_lines:
    pts = [(v.get("x", 0), v.get("y", 0)) for v in line["bbox"]]
    pts = [(int(x), int(y)) for x, y in pts]

    # Draw contour
    cv2.polylines(img, [cv2.convexHull(np.array(pts))], True, (0, 255, 0), 2)

    # Draw text above box
    x, y = pts[0]
    cv2.putText(img, line["text"], (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


output_path = "output_annotated.jpg"
cv2.imwrite(output_path, img)
print(f"✅ Output saved as {output_path}")

with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(paragraphs))
print("✅ Extracted text saved as extracted_text.txt")
