import os
import shutil
import random

images_dir = "images"
labels_dir = "labels"
output_dir = "dataset"

os.makedirs(f"{output_dir}/train/fire", exist_ok=True)
os.makedirs(f"{output_dir}/train/no_fire", exist_ok=True)
os.makedirs(f"{output_dir}/val/fire", exist_ok=True)
os.makedirs(f"{output_dir}/val/no_fire", exist_ok=True)

images = os.listdir(images_dir)
random.shuffle(images)

split = int(0.8 * len(images))
train_images = images[:split]
val_images = images[split:]

def process_images(image_list, split_type):
    for img_name in image_list:
        label_name = img_name.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_name)

        image_path = os.path.join(images_dir, img_name)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                content = f.read().strip()

            if content:  # если есть объекты → fire
                target_class = "fire"
            else:
                target_class = "no_fire"
        else:
            target_class = "no_fire"

        dest_path = os.path.join(output_dir, split_type, target_class, img_name)
        shutil.copy(image_path, dest_path)

process_images(train_images, "train")
process_images(val_images, "val")

print("Датасет подготовлен!")
