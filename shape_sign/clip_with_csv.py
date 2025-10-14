import os
import csv
import torch
import clip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from collections import Counter, defaultdict
from torchvision import transforms

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, preprocess = clip.load("ViT-B/32", device=device)

# === PATHS ===
image_folder = r"C:\Users\ASUS\Downloads\img_rename"
output_folder = r"C:\Users\ASUS\Downloads\CLIP_Labeled_Outputs"
csv_path = os.path.join(output_folder, "output.csv")

os.makedirs(output_folder, exist_ok=True)
if os.path.exists(csv_path):
    os.remove(csv_path)

# === FONT ===
try:
    font = ImageFont.truetype("arial.ttf", 38)
except:
    font = ImageFont.load_default()

# === EXPANDED PROMPT GROUPS (prompt engineering) ===
prompt_groups = {
    "signature_on_paper": [
        "a handwritten signature on white paper",
        "a black ink signature on a document",
        "a person's signature at the bottom of a paper",
        "a scanned image of a signed paper",
        "a signature drawn with a pen on a page",
        "a handwritten signature in blue ink on paper",
        "a paper containing someone's signature",
        "a signed document on a desk",
        "a close-up of a signature written on white sheet",
        "a written signature on paper"
    ],
    "shape_on_device": [
        "a geometric shape displayed on a smartphone screen",
        "a digital drawing of a shape on a tablet",
        "a shape shown on a mobile device",
        "a circle or square displayed on a phone screen",
        "a digital shape rendered on an electronic device",
        "a pattern or shape shown on a touch screen",
        "a geometric icon on a phone display",
        "a shape drawn on a digital device",
        "a simple geometric design on a tablet",
        "a mobile screen showing a shape"
    ],
    "shape_on_paper": [
        "a geometric shape drawn with pen on paper",
        "a simple circle or triangle on white paper",
        "a hand-drawn shape on a notebook page",
        "a pencil sketch of a shape on paper",
        "a paper with a shape drawn on it",
        "a diagram of a shape on a sheet of paper",
        "a rectangular or circular figure drawn on paper",
        "a geometry shape drawn with pencil on a sheet",
        "a photo of paper showing a shape",
        "a shape drawn by hand on white paper"
    ],
    "signature_on_device": [
        "a digital signature on a mobile screen",
        "a stylus drawing a signature on a tablet",
        "an electronic signature displayed on a touchscreen",
        "a person signing digitally on a phone",
        "a digital handwritten signature on device screen",
        "a screenshot of a signature on a tablet",
        "a stylus-based signature on mobile device",
        "a finger-written signature on touchscreen",
        "a signed document displayed on tablet",
        "a photo of a digital signature on phone"
    ],
    "noisy_data": [
        "a blurry or corrupted image",
        "a noisy, distorted, or unclear picture",
        "an image with heavy noise and low clarity",
        "a broken or pixelated visual",
        "a corrupted or unreadable image file",
        "an unclear or partially damaged image",
        "a distorted visual artifact",
        "an image with random patterns or pixels",
        "a glitched or low-quality image",
        "a visually noisy and unclear picture"
    ]
}

# === PREPARE PROMPT EMBEDDINGS ===
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

# === PREFIX MAP ===
label_prefix = {
    "signature_on_device": "sod",
    "signature_on_paper": "sop",
    "shape_on_device": "shod",
    "shape_on_paper": "shop",
    "noisy_data": "noise"
    
}

label_counters = defaultdict(int)

# === CSV FUNCTION ===
def log_result(writer, filename, predicted, confidence, remark="", actual=""):
    formatted_conf = f"{confidence:.2f}" if confidence is not None else ""
    writer.writerow([filename, predicted, formatted_conf, remark, actual])

# === IMAGE AUGMENTATION (light jitter for robustness) ===
augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(degrees=3)
])

# === MAIN LOOP ===
print("Starting classification...")

valid_exts = ('.png', '.jpg', '.jpeg', '.jfif', '.bmp', '.webp', '.tiff')

# Load optional ground truth (if exists)
ground_truth_map = {}
gt_path = os.path.join(image_folder, "ground_truth.csv")
if os.path.exists(gt_path):
    with open(gt_path, newline='', encoding='utf-8') as gtfile:
        reader = csv.DictReader(gtfile)
        for row in reader:
            ground_truth_map[row["filename"]] = row.get("actual_result", "")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "predicted_output", "confidence", "remark", "actual_result"])

    for root, _, files in os.walk(image_folder):
        for filename in sorted(files):
            if not filename.lower().endswith(valid_exts):
                continue

            img_path = os.path.join(root, filename)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                log_result(writer, filename, "Unreadable", None, f"failed: {e}")
                continue

            try:
                image = ImageEnhance.Contrast(image).enhance(1.5)
                image_aug = augment(image)

                # Encode image
                image_tensor = preprocess(image_aug).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = image_features @ text_features.T

                # Confidence and label
                # Compute similarity and normalized confidence
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                best_idx = similarity.argmax().item()
                avg_conf = similarity[0][best_idx].item()
                final_label = class_labels[best_idx]

            
                # Draw label
                draw = ImageDraw.Draw(image)
                label_text = f"{final_label} ({avg_conf*100:.1f}%)"
                draw.rectangle([0, 0, image.width, 40], fill=(0, 0, 0))
                draw.text((10, 5), label_text, fill=(255, 255, 255), font=font)

                # Filename logic
                label_counters["global"] += 1
                new_filename = f"img_{label_counters['global']}.jpg"

                save_path = os.path.join(output_folder, new_filename)
                image.save(save_path)

                # Compare with ground truth (if available)
                actual = ground_truth_map.get(filename, "")
                remark = ""
                if actual and actual != final_label:
                    remark = "Mismatch"

                # Log
                log_result(writer, new_filename, final_label, avg_conf, remark, actual)

            except Exception as e:
                log_result(writer, filename, "Error", None, f"failed: {e}")

print(f"\n All labeled images saved in: {output_folder}")
print(f" CSV log saved at: {csv_path}")
