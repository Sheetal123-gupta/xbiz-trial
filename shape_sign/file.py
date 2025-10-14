import os
import shutil

source_folder = r"C:\Users\ASUS\Downloads\Sign_samples_all"
destination_folder = r"C:\Users\ASUS\Downloads\img_rename"
os.makedirs(destination_folder, exist_ok=True)

image_extensions=('.jpg','.jpeg','.png','.bmp','.gif','.webp')
for idx,filename in enumerate(os.listdir(source_folder),start=0):
  if filename.lower().endswith(image_extensions):
    new_name=f'img_{idx}'+os.path.splitext(filename)[1]
    src_path=os.path.join(source_folder,filename)
    dst_path=os.path.join(destination_folder,new_name)
    shutil.copy2(src_path,dst_path)
print('chalo saved ')