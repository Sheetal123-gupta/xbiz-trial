import os
import csv
import torch
import clip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from collections import Counter

# === CONFIG (GPU/CPU selection) ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === MODEL LOAD ===
model, preprocess = clip.load("ViT-B/32", device=device)

# === PATHS ===
image_folder = r"C:\Users\ASUS\Downloads\Sign_samples_all"
output_folder = r"C:\Users\ASUS\Downloads\CLIP_Labeled_Output4"
csv_path = os.path.join(output_folder, "output.csv")

os.makedirs(output_folder, exist_ok=True)

# === RESET CSV IF EXISTS ===
if os.path.exists(csv_path):
    os.remove(csv_path)

# === FONT ===
try:
    font = ImageFont.truetype("arial.ttf", 38)
except:
    font = ImageFont.load_default()

# === PROMPT GROUPS ===
prompt_groups = {
    "signature_on_paper": [
        "a handwritten signature on white paper",
        "a person's signature on a sheet"
    ],
    "shape_on_device": [
        "a geometric shape on a mobile screen",
        "a digital shape on a tablet"
    ],
    "shape_on_paper": [
        "a shape drawn on paper"
    ],
    "signature_on_device": [
        "a digital signature on a mobile device"
    ],
    "noisy_data": [
        "a blurry or corrupted image",
        "a noisy or distorted visual",
        "an unclear or broken image"
    ]
}

# === ENCODE TEXT PROMPTS ===
class_labels = []
text_features = []

for label, variants in prompt_groups.items():
    tokens = clip.tokenize(variants).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        mean_embedding = embeddings.mean(dim=0, keepdim=True)
        text_features.append(mean_embedding)
        class_labels.append(label)

text_features = torch.cat(text_features, dim=0)

# === LOG FUNCTION ===
def log_result(csv_path, filename, label, confidence=None, status="success", remark=""):
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["filename", "label", "confidence", "status", "remark"])
            formatted_confidence = f"{confidence:.2f}" if confidence is not None else ""
            writer.writerow([filename, label, formatted_confidence, status, remark])
    except PermissionError:
        print(f" Permission denied: Close {csv_path} if open in Excel.")

# === PROCESS IMAGES ===
print("Starting classification...")

valid_exts = ('.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.webp', '.tiff')

for root, _, files in os.walk(image_folder):
    for filename in files:
        if not filename.lower().endswith(valid_exts):
            continue

        img_path = os.path.join(root, filename)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log_result(csv_path, filename, "Unreadable", None, "failed", str(e))
            continue

        try:
            image = ImageEnhance.Contrast(image).enhance(1.5)

            votes = []
            confidences = []

            for _ in range(3):  # ensemble runs
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    best_idx = similarity.argmax().item()
                    votes.append(class_labels[best_idx])
                    confidences.append(similarity[0][best_idx].item())

            vote_counts = Counter(votes)
            final_label, _ = vote_counts.most_common(1)[0]
            avg_conf = sum(confidences) / len(confidences)

            THRESHOLD = 0.45
            label_to_draw = final_label if avg_conf >= THRESHOLD else "Unknown"

            # Draw label
            draw = ImageDraw.Draw(image)
            label_text = f"{label_to_draw} ({avg_conf*100:.1f}%)"
            draw.rectangle([0, 0, image.width, 40], fill=(0, 0, 0))
            draw.text((10, 5), label_text, fill=(255, 255, 255), font=font)

            # Save labeled image
            save_path = os.path.join(output_folder, filename)
            image.save(save_path)

            # Log result
            log_result(csv_path, filename, label_to_draw, avg_conf)

        except Exception as e:
            log_result(csv_path, filename, "Error", None, "failed", str(e))

print(f"\n All images processed. Labeled images saved in: {output_folder}")
print(f" CSV log saved at: {csv_path}")
