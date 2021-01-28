import json
import os
import sys

import cv2
import imageio

all_products_file = sys.argv[1]
dataset_root = sys.argv[2]
dataset_target_root = sys.argv[3]
dataset_target_items = int(sys.argv[4])

with open(all_products_file, "r") as f:
    all_products = json.load(f)

products_tops = {
    key: value for key, value in all_products.items()
    if value["tryon"]["category"] == "tops"
}

model_photo_dir = os.path.join(dataset_target_root, "train_img")
cloth_photo_dir = os.path.join(dataset_target_root, "train_color")
os.makedirs(model_photo_dir, exist_ok=True)
os.makedirs(cloth_photo_dir, exist_ok=True)

paths_map = {}

counter = 0
for key, item in products_tops.items():
    item_media = item["media_metadata"][0]
    category = str(item.get("tryon", {}).get("category", "__no_tryon"))

    src_dataset_dir = os.path.join(dataset_root, category, key, item_media["id"])

    cloth_image_id = item_media["display_images_order"][0]
    cloth_image_url = item_media["display_images"][cloth_image_id]
    cloth_image_path = os.path.join(src_dataset_dir, f"{cloth_image_id}_{os.path.split(cloth_image_url)[-1]}")
    cloth_image = imageio.imread(cloth_image_path)
    cloth_image = cv2.resize(cloth_image, dsize=(192, 256), interpolation=cv2.INTER_AREA)

    person_image_1_id = item_media["display_images_order"][1]
    person_image_1_url = item_media["display_images"][person_image_1_id]
    person_image_1_path = os.path.join(src_dataset_dir, f"{person_image_1_id}_{os.path.split(person_image_1_url)[-1]}")
    person_image_1 = imageio.imread(person_image_1_path)
    person_image_1 = cv2.resize(person_image_1, dsize=(192, 256), interpolation=cv2.INTER_AREA)
    imageio.imwrite(os.path.join(model_photo_dir, f"{counter}_0.jpg"), person_image_1, quality=97)
    imageio.imwrite(os.path.join(cloth_photo_dir, f"{counter}_0.jpg"), cloth_image, quality=97)

    paths_map[key] = {
        "0": {
            "cloth_path_src": cloth_image_path,
            "person_image_path": person_image_1_path,
            "cloth_path_dst": os.path.join(cloth_photo_dir, f"{counter}_0.jpg"),
            "person_path_dst": os.path.join(model_photo_dir, f"{counter}_0.jpg"),
        }
    }

    if len(item_media["display_images_order"]) > 2:
        person_image_2_id = item_media["display_images_order"][2]
        person_image_2_url = item_media["display_images"][person_image_2_id]
        person_image_2_path = os.path.join(src_dataset_dir, f"{person_image_2_id}_{os.path.split(person_image_2_url)[-1]}")
        person_image_2 = imageio.imread(person_image_2_path)
        person_image_2 = cv2.resize(person_image_2, dsize=(192, 256), interpolation=cv2.INTER_AREA)
        imageio.imwrite(os.path.join(model_photo_dir, f"{counter}_1.jpg"), person_image_2, quality=97)
        imageio.imwrite(os.path.join(cloth_photo_dir, f"{counter}_1.jpg"), cloth_image, quality=97)

        paths_map[key]["1"] = {
            "cloth_path_src": cloth_image_path,
            "person_image_path": person_image_1_path,
            "cloth_path_dst": os.path.join(cloth_photo_dir, f"{counter}_1.jpg"),
            "person_path_dst": os.path.join(model_photo_dir, f"{counter}_1.jpg"),
        }

    counter += 1
    if counter > dataset_target_items:
        break

with open("dataset_paths_map.json", "w") as f:
    json.dump(paths_map, f)
