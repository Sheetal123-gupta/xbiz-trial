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
import json

def extract_text_from_response(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_text = []

    try:
        responses = data["results"][0]["Data"]["responses"]
        for response in responses:
            full_text = response.get("fullTextAnnotation", {})
            for page in full_text.get("pages", []):
                for block in page.get("blocks", []):
                    block_text = []
                    for para in block.get("paragraphs", []):
                        para_text = []
                        for word in para.get("words", []):
                            word_text = "".join([s["text"] for s in word.get("symbols", [])])
                            para_text.append(word_text)
                        block_text.append(" ".join(para_text))
                    output_text.append("\n".join(block_text))
    except Exception as e:
        print("Error parsing:", e)

    return "\n".join(output_text)

# Usage
text = extract_text_from_response("response.json")
print("==== Extracted OCR Text ====")
print(text)
