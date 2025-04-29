import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

input_dir = r"C:\Users\anisa\AppData\Local\Programs\Dataset"
output_dir = r"C:\Users\anisa\AppData\Local\Programs\Augmen data"

def augment_image(image):
    augmented = []

    # Original
    augmented.append(('original', image.copy()))

    # Mirrored
    mirrored = cv2.flip(image, 1)
    augmented.append(('mirrored', mirrored))

    # Flipped vertically
    flipped = cv2.flip(image, 0)
    augmented.append(('flipped', flipped))

    return augmented

for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for idx, file_name in enumerate(os.listdir(class_path)):
        file_path = os.path.join(class_path, file_name)
        image = cv2.imread(file_path)
        if image is None:
            continue

        augmentations = augment_image(image)

        for aug_type, aug_img in augmentations:
            save_path = os.path.join(output_class_dir, f"{class_name}_{idx}_{aug_type}.jpg")
            cv2.imwrite(save_path, aug_img)

print("✅ Advanced augmentation completed!")
