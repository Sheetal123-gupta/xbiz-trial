'''
import os
import csv

# === CONFIG ===
DATASET_DIR = "C:\\Users\\ASUS\\Downloads\\dataset"
OUTPUT_CSV = "labels.csv"

def collect_image_labels(root_dir):
    data = []

    # Handle 'signature' folder
    sig_dir = os.path.join(root_dir, "signature")
    if os.path.exists(sig_dir):
        for filename in os.listdir(sig_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join("signature", filename)
                data.append([filepath, "signature"])

    # Handle 'shapes' subfolders
    shapes_dir = os.path.join(root_dir, "shapes")
    if os.path.exists(shapes_dir):
        for shape in os.listdir(shapes_dir):
            shape_path = os.path.join(shapes_dir, shape)
            if os.path.isdir(shape_path):
                for filename in os.listdir(shape_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepath = os.path.join("shapes", shape, filename)
                        data.append([filepath, shape])

    return data

def write_csv(data, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])
        writer.writerows(data)
    print(f"✅ CSV created: {output_csv} with {len(data)} entries")

# === RUN ===
image_data = collect_image_labels(DATASET_DIR)
write_csv(image_data, OUTPUT_CSV)
'''

import os
import torch
import clip
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Path to your local folder containing images
image_folder = r"C:\Users\ASUS\Downloads\Sign_samples_all"  # 👈 change this
output_folder = r"C:\Users\ASUS\Downloads\CLIP_Labeled_Output"

# Create output folder if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# === PROMPTS ===
prompts = [
    "a shape on a device",
    "a signature on paper",
    "a shape on paper",
    "a shape on a mobile screen",
    "some noisy data"
]
text = clip.tokenize(prompts).to(device)

# === FONT (optional) ===
try:
    font = ImageFont.truetype("arial.ttf", 28)  # Windows default font
except:
    font = ImageFont.load_default()

print("🔍 Starting classification...")

# === LOOP THROUGH IMAGES ===
for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, filename)

            try:
                # Preprocess
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

                # Encode image + text
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    best_idx = similarity.argmax().item()
                    confidence = similarity[0][best_idx].item()

                # === DRAW LABEL ===
                draw = ImageDraw.Draw(image)
                label = f"{prompts[best_idx]} ({confidence*100:.1f}%)"
                draw.rectangle([0, 0, image.width, 40], fill=(0, 0, 0))
                draw.text((10, 5), label, fill=(255, 255, 255), font=font)

                # Save labeled image
                save_path = os.path.join(output_folder, filename)
                image.save(save_path)

                #print(f"✅ {filename}: {label}")

            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

print(f"\n🎉 All labeled images saved in: {output_folder}")
