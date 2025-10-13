#clip works on semantic similarity between the image and text prompts 
import os
import torch
import clip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from collections import Counter

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = r"C:\Users\ASUS\Downloads\Sign_samples_all"
output_folder = r"C:\Users\ASUS\Downloads\CLIP_Labeled_Output3"
os.makedirs(output_folder, exist_ok=True)

# === FONT ===
try:
    font = ImageFont.truetype("arial.ttf", 38)
except:
    font = ImageFont.load_default()

# === PROMPT GROUPS (well-separated) ===
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
    "signature_on_device":[
        "Signed on paper"
    ],
    "noisy_data": [
        "a blurry or corrupted image",
        "a noisy or distorted visual",
        "an unclear or broken image"
    ]
}

# === ENCODE PROMPT ENSEMBLES ===
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

# === LOOP THROUGH IMAGES ===
print(" Starting classification...")

for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, filename)

            try:
                image = Image.open(img_path).convert("RGB")
                image = ImageEnhance.Contrast(image).enhance(1.5)

                votes = []
                confidences = []

                # === MULTIPLE RUNS (ensemble voting) ===
                for _ in range(3):  # run 3 times
                    image_tensor = preprocess(image).unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image_tensor)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                        best_idx = similarity.argmax().item()
                        votes.append(class_labels[best_idx])
                        confidences.append(similarity[0][best_idx].item())

                # === VOTING + THRESHOLDING ===
                vote_counts = Counter(votes)
                final_label, vote_freq = vote_counts.most_common(1)[0]
                avg_conf = sum(confidences) / len(confidences)

                THRESHOLD = 0.45
                label_to_draw = final_label if avg_conf >= THRESHOLD else "Unknown"

                # === DRAW LABEL ===
                draw = ImageDraw.Draw(image)
                label_text = f"{label_to_draw} ({avg_conf*100:.1f}%)"
                draw.rectangle([0, 0, image.width, 40], fill=(0, 0, 0))
                draw.text((10, 5), label_text, fill=(255, 255, 255), font=font)

                # === SAVE IMAGE ===
                save_path = os.path.join(output_folder, filename)
                image.save(save_path)

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

print(f"\n All labeled images saved in: {output_folder}")
