'''
Curl Request
curl --location 'https://bankdevapi.digivision.ai/digivision/ai/rawtext-extraction' \
--header 'Content-Type: application/json' \
--data '{
    "txnId": "TXN0000001",
    "docType": ".JPG",
    "source": "OCR_RAW",
    "documentName": "14.JPG",
    "caseNo": "case001",
    "documentBlob": "--iske undar apna base64 ka wo dalna he"
}
'''
#OCR JSON processing & visualization 

import json

# Load OCR JSON
with open("ocr_response.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lines = []

# Traverse OCR response
for page_idx, page in enumerate(data["results"][0]["Data"]["responses"][0]["fullTextAnnotation"]["pages"]):
    for block_idx, block in enumerate(page["blocks"]):
        for para_idx, para in enumerate(block["paragraphs"]):
            para_text = ""
            for word in para["words"]:
                word_text = "".join([s["text"] for s in word["symbols"]])
                para_text += word_text + " "
            para_text = para_text.strip()

            if para_text:
                # Take top-left Y and X for sorting
                y_coord = para["boundingBox"]["vertices"][0].get("y", 0)
                x_coord = para["boundingBox"]["vertices"][0].get("x", 0)

                lines.append({
                    "text": para_text,
                    "y": y_coord,
                    "x": x_coord
                })

# Sort lines top → bottom, then left → right
sorted_lines = sorted(lines, key=lambda x: (x["y"], x["x"]))

# Extract text in order
final_text_lines = [line["text"] for line in sorted_lines]

# Save to file
with open("aligned_text.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_text_lines))

print("✅ Aligned text saved to aligned_text.txt\n")
print("\n".join(final_text_lines))
