import json
import os
import sys

import cv2
import imageio
import numpy

input_path = sys.argv[1]
output_path = sys.argv[2]
photos_path = sys.argv[3]

if not os.path.exists(input_path):
    print(f"{input_path} does not exist")

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

os.makedirs(os.path.join(output_path, "test_reference"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_reference_crop"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_color"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_img"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_img_ref"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_photo_model"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_photo_model_src"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test_photo_cloth_src"), exist_ok=True)

test_idx = 0
for dirname in sorted(os.listdir(input_path)):
    variants_list = sorted(os.listdir(os.path.join(input_path, dirname)))
    reference, tests = variants_list[0], variants_list[1:]

    with open(os.path.join(input_path, dirname, reference, "meta.json"), "r") as f:
        reference_meta = json.load(f)
    model_id = reference_meta["model_id"]
    top_product_id = dict(zip(reference_meta["product_categories"], reference_meta["product_ids"]))["tops"]

    reference_model_img = imageio.imread(os.path.join(input_path, dirname, reference, "model_file.png"))
    h, w = reference_model_img.shape[:2]
    reference_model_img_crop = reference_model_img[:256, (w - 192) // 2 : -(w - 192) // 2]

    model_not_found = True
    for category in os.listdir(photos_path):
        reference_model_photo_dir = os.path.join(photos_path, category, model_id + "_0", model_id)
        try:
            reference_model_photo_filename = sorted(os.listdir(reference_model_photo_dir))[1]
        except FileNotFoundError:
            continue
        else:
            reference_model_photo = imageio.imread(
                os.path.join(reference_model_photo_dir, reference_model_photo_filename)
            )
            h, w = reference_model_photo.shape[:2]
            scale_factor = 512 / h
            reference_model_photo = cv2.resize(
                reference_model_photo, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
            )
            h, w = reference_model_photo.shape[:2]
            reference_model_photo_crop = reference_model_photo[:256, (w - 192) // 2 : -(w - 192) // 2]
            model_not_found = False
            break
    if model_not_found:
        print(f"Failed to find photo of model {model_id}")
        reference_model_photo = None
        reference_model_photo_crop = numpy.ones(shape=(256, 192, 3), dtype=numpy.uint8) * 200

    for test in tests:
        imageio.imwrite(os.path.join(output_path, "test_img_ref", f"{test_idx}.jpg"), reference_model_img, quality=97)
        imageio.imwrite(os.path.join(output_path, "test_img", f"{test_idx}.jpg"), reference_model_img_crop, quality=97)

        if reference_model_photo is not None:
            imageio.imwrite(
                os.path.join(output_path, "test_photo_model_src", f"{test_idx}.jpg"), reference_model_photo, quality=97
            )
        imageio.imwrite(
            os.path.join(output_path, "test_photo_model", f"{test_idx}.jpg"), reference_model_photo_crop, quality=97
        )

        with open(os.path.join(input_path, dirname, test, "meta.json"), "r") as f:
            reference_meta = json.load(f)
        top_product_id = dict(zip(reference_meta["product_categories"], reference_meta["product_ids"]))["tops"]

        reference_cloth_photo_dir = os.path.join(photos_path, "tops", top_product_id, top_product_id[:-2])
        try:
            reference_cloth_photo_filename = sorted(os.listdir(reference_cloth_photo_dir))[1]
        except FileNotFoundError:
            print(f"{reference_model_photo_dir} does not exist")
            reference_cloth_photo = None
            reference_cloth_photo_crop = numpy.ones(shape=(256, 192, 3), dtype=numpy.uint8) * 200
        else:
            reference_cloth_photo = imageio.imread(
                os.path.join(reference_cloth_photo_dir, reference_cloth_photo_filename)
            )
            h, w = reference_cloth_photo.shape[:2]
            scale_factor = 512 / h
            reference_cloth_photo = cv2.resize(
                reference_cloth_photo, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
            )
            h, w = reference_cloth_photo.shape[:2]
            reference_cloth_photo_crop = reference_cloth_photo[:256, (w - 192) // 2 : -(w - 192) // 2]
        if reference_cloth_photo is not None:
            imageio.imwrite(
                os.path.join(output_path, "test_photo_cloth_src", f"{test_idx}.jpg"), reference_cloth_photo, quality=97
            )
        imageio.imwrite(
            os.path.join(output_path, "test_photo_cloth", f"{test_idx}.jpg"), reference_cloth_photo_crop, quality=97
        )

        img = imageio.imread(os.path.join(input_path, dirname, test, "tops.jpg"))
        img = cv2.resize(img, dsize=(192, 256), interpolation=cv2.INTER_AREA)
        imageio.imwrite(os.path.join(output_path, "test_color", f"{test_idx}.jpg"), img, quality=97)

        img = imageio.imread(os.path.join(input_path, dirname, test, "model_file.png"))
        imageio.imwrite(os.path.join(output_path, "test_reference", f"{test_idx}.jpg"), img, quality=97)
        h, w = img.shape[:2]
        img_crop = img[:256, (w - 192) // 2 : -(w - 192) // 2]
        imageio.imwrite(os.path.join(output_path, "test_reference_crop", f"{test_idx}.jpg"), img_crop, quality=97)

        test_idx += 1
