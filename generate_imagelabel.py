import os
import numpy as np
from PIL import Image

def process_image_folder(folder_path, output_npy_path):
    label_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("L")

            image_array = np.array(image)

            white_pixel_count = np.sum(image_array > 0)

            total_pixels = image_array.size

            white_pixel_percentage = (white_pixel_count / total_pixels) * 100

            # If the percentage of white pixels is higher than 0.30%, the value is 1; otherwise, it is 0
            label_value = 1 if white_pixel_percentage >= 0.30 else 0

            label_dict[filename] = np.array([label_value], dtype=np.float32)

    np.save(output_npy_path, label_dict)
    print(f"Labels saved to {output_npy_path}")

folder_path = "E:\working\Data\label"
output_npy_path = r"E:\working\python\datasets\CLCD/imagelevel_labels.npy"
process_image_folder(folder_path, output_npy_path)