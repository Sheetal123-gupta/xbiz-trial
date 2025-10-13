import os
import shutil

source_folder = r"C:\Users\ASUS\Downloads\Sign_samples_all"
destination_folder = r"C:\Users\ASUS\Downloads\Renamed_Files"
os.makedirs(destination_folder, exist_ok=True)

counter = 1

for root, dirs, files in os.walk(source_folder):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            old_path = os.path.join(root, filename)
            new_filename = f"img_{counter:03d}" + os.path.splitext(filename)[1]
            new_path = os.path.join(destination_folder, new_filename)

            shutil.copy2(old_path, new_path)  # preserves metadata
            counter += 1

print(f"✅ Renamed and copied {counter - 1} files to: {destination_folder}")
