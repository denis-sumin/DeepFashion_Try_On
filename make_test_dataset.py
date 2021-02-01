import os
import shutil
import sys

import cv2
import imageio

input_path = sys.argv[1]
output_path = sys.argv[2]

if not os.path.exists(input_path):
    print(f"{input_path} does not exist")

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

os.makedirs(os.path.join(output_path, "test_reference"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_reference_crop"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_color"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_img"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_img_ref"), exist_ok=True)

test_idx = 0
for dirname in os.listdir(input_path):
    variants_list = os.listdir(os.path.join(input_path, dirname))
    reference, tests = variants_list[0], variants_list[1:]

    for test in tests:
        img = imageio.imread(os.path.join(input_path, dirname, reference, "model_file.png"))
        imageio.imwrite(os.path.join(output_path, "test_img_ref", f"{test_idx}.jpg"), img, quality=97)
        h, w = img.shape[:2]
        img_crop = img[:256, (w-192) // 2:-(w-192) // 2]
        imageio.imwrite(os.path.join(output_path, "test_img", f"{test_idx}.jpg"), img_crop, quality=97)
        img = imageio.imread(os.path.join(input_path, dirname, test, "tops.jpg"))
        img = cv2.resize(img, dsize=(192, 256), interpolation=cv2.INTER_AREA)
        imageio.imwrite(os.path.join(output_path, "test_color", f"{test_idx}.jpg"), img, quality=97)
        img = imageio.imread(os.path.join(input_path, dirname, test, "model_file.png"))
        imageio.imwrite(os.path.join(output_path, "test_reference", f"{test_idx}.jpg"), img, quality=97)
        h, w = img.shape[:2]
        img_crop = img[:256, (w-192) // 2:-(w-192) // 2]
        imageio.imwrite(os.path.join(output_path, "test_reference_crop", f"{test_idx}.jpg"), img_crop, quality=97)
        test_idx += 1
