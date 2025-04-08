import os
import random
import shutil
from PIL import Image

# ตั้งค่าพาธโฟลเดอร์
input_folder = "r_frames"
output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

# สร้างโฟลเดอร์ Train, Validation, Test
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
test_dir = os.path.join(output_folder, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# ดึงรายการไฟล์ทั้งหมด
image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

# สุ่มแบ่งข้อมูล
random.shuffle(image_files)
num_train = int(0.8 * len(image_files))
num_val = int(0.1 * len(image_files))

train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# ฟังก์ชัน Resize และบันทึกไฟล์
def process_and_save(files, target_dir):
    for file in files:
        img = Image.open(os.path.join(input_folder, file))
        img = img.resize((112, 112))  # Resize เป็น 112x112
        img.save(os.path.join(target_dir, file))

# บันทึกไฟล์ที่ Resize แล้วลงโฟลเดอร์ต่างๆ
process_and_save(train_files, train_dir)
process_and_save(val_files, val_dir)
process_and_save(test_files, test_dir)

print("✅ จัดการ Dataset เรียบร้อย!")
