import json
import os
import sys

import cv2
import imageio
import numpy

data_dir = sys.argv[1]
data_prefix = sys.argv[2]

lip_labels_map = {
    0: 0,  # Background
    2: 1,  # Hair
    5: 4,  # Upper-clothes
    9: 8,  # Pants
    13: 12,  # Face
    14: 11,  # Left-arm
    15: 13,  # Right-arm
    16: 9,  # Left-leg
    17: 10,  # Right-leg
    18: 5,  # Left-shoe
    19: 6,  # Right-shoe
    # additional
    1: 7,  # Hat -> Noise
    4: 12,  # Sunglasses -> Face
    3: 7,  # Glove -> Noise
    6: 4,  # Dress -> Upper-clothes
    7: 4,  # Coat -> Upper-clothes
    8: 7,  # Socks -> Noise
    10: 4,  # Jumpsuits -> Upper-clothes
    11: 7,  # Scarf -> Noise
    12: 8,  # Skirt -> Pants
}
label_src_images_dir = os.path.join(data_dir, data_prefix + "_label_src")
label_images_dir = os.path.join(data_dir, data_prefix + "_label")
os.makedirs(label_images_dir, exist_ok=True)
for filename in sorted(os.listdir(label_src_images_dir)):
    if not filename.endswith("png"):
        continue
    src_image_path = os.path.join(label_src_images_dir, filename)
    dst_image_path = os.path.join(label_images_dir, filename)
    image = imageio.imread(src_image_path)
    image_new = numpy.zeros(shape=image.shape, dtype=image.dtype)
    for key, value in lip_labels_map.items():
        image_new[image == key] = value
    imageio.imwrite(dst_image_path, image_new)

cloth_images_dir = os.path.join(data_dir, data_prefix + "_color")
edge_images_dir = os.path.join(data_dir, data_prefix + "_edge")
os.makedirs(edge_images_dir, exist_ok=True)
for filename in sorted(os.listdir(cloth_images_dir)):
    if filename[-3:] not in ("jpg", "png"):
        continue

    # TODO: results on very white clothes are not good
    # if filename != "000164_1.jpg":
    #     continue

    src_image_path = os.path.join(cloth_images_dir, filename)
    dst_image_path = os.path.join(edge_images_dir, filename)
    src_image = imageio.imread(src_image_path)

    gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)

    ret, mask_first = cv2.threshold(gray, 254, 1, cv2.THRESH_BINARY_INV)

    # gray[mask_first.astype(numpy.bool)] = gray[mask_first.astype(numpy.bool)].astype(numpy.float) * 0.98
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.6)
    gray = cv2.GaussianBlur(gray, (3, 3), sigmaX=1.6)

    # b, g, r = cv2.split(src_image)
    threshold = 250

    # ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 11, 2)
    ret, mask = cv2.threshold(gray, threshold, 1, cv2.THRESH_BINARY_INV)

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se2)
    mask = cv2.erode(mask, se1)

    output_mask = mask.astype("uint8") * 255

    imageio.imwrite(dst_image_path, output_mask)

pose_src_images_dir = os.path.join(data_dir, data_prefix + "_pose_src")
pose_images_dir = os.path.join(data_dir, data_prefix + "_pose")
os.makedirs(pose_images_dir, exist_ok=True)
for filename in sorted(os.listdir(pose_src_images_dir)):
    if not filename.endswith("json"):
        continue
    src_file_path = os.path.join(pose_src_images_dir, filename)
    dst_file_path = os.path.join(pose_images_dir, filename)
    with open(src_file_path, "r") as f:
        pose_data = json.load(f)

    pose_data["people"][0]["pose_keypoints"] = pose_data["people"][0].pop("pose_keypoints_2d")

    with open(dst_file_path, "w") as f:
        json.dump(pose_data, f)
