import argparse
import json
import multiprocessing
import os
import subprocess
from functools import partial
from typing import Callable, Sequence, Tuple, Set

import cv2
import imageio
import numpy


def run_openpose(source_images_dir: str, dst_dir: str):
    cwd = "/root/openpose"
    env = {
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    cmd = [
        "./build/examples/openpose/openpose.bin",
        "-image_dir", source_images_dir,
        "-write_images", dst_dir,
        "-display", "0",
        "--cli_verbose", "0.01",
        "-write_json", dst_dir,
        "-model_pose", "COCO"
    ]
    subprocess.run(args=cmd, cwd=cwd, env=env)


def run_segmentation(source_images_dir: str, dst_dir: str):
    cwd = "/root/Self-Correction-Human-Parsing"
    env = {
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "PATH": "/usr/local/bin:/usr/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda-10.0/lib64"
    }
    cmd = [
        "./venv/bin/python", "simple_extractor.py",
        "--dataset", "lip",
        "--model-restore", "checkpoints/lip.pth",
        "--input-dir", source_images_dir,
        "--output-dir", dst_dir,
    ]
    subprocess.run(args=cmd, cwd=cwd, env=env)


def prepare_one_cloth_mask(src_path: str, dst_path: str) -> None:
    # TODO: results on very white clothes are not good
    # if filename != "000164_1.jpg":
    #     continue

    src_image = imageio.imread(src_path)

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

    imageio.imwrite(dst_path, output_mask)


def process_one_label_file(src_path: str, dst_path: str) -> None:
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
        1: 1,  # Hat -> Hair
        4: 12,  # Sunglasses -> Face
        3: 7,  # Glove -> Noise
        6: 4,  # Dress -> Upper-clothes
        7: 4,  # Coat -> Upper-clothes
        8: 7,  # Socks -> Noise
        10: 7,  # Jumpsuits -> Upper-clothes
        11: 7,  # Scarf -> Noise
        12: 8,  # Skirt -> Pants
    }

    image = imageio.imread(src_path)
    image_new = numpy.zeros(shape=image.shape, dtype=image.dtype)
    for key, value in lip_labels_map.items():
        image_new[image == key] = value
    imageio.imwrite(dst_path, image_new)


def process_one_pose_file(src_path: str, dst_path: str) -> None:
    with open(src_path, "r") as f:
        pose_data = json.load(f)

    try:
        pose_data["people"][0]["pose_keypoints"] = pose_data["people"][0].pop("pose_keypoints_2d")
    except IndexError:
        print(f"IndexError: pose_data['people'][0] element doesn't exist in {src_path}")
    except KeyError as e:
        print(f"KeyError: {e} in {src_path}")
    else:
        with open(dst_path, "w") as f:
            json.dump(pose_data, f)


def process_one_file_star(func: Callable, arg: Tuple[str, str]) -> None:
    try:
        return func(*arg)
    except Exception as e:
        print(f"{type(e)}: {e}. {func}, arguments: {arg}")


def process_many_files(function: Callable, src_dir: str, dst_dir: str,
                       src_extensions: Sequence[str], dst_extension: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    filenames = [
        (os.path.join(src_dir, filename),
         os.path.join(dst_dir, os.path.splitext(filename)[0] + dst_extension))
        for filename in sorted(os.listdir(src_dir))
        if os.path.splitext(filename)[-1] in src_extensions
    ]

    items_count = len(filenames)
    print_every = max(1, int(0.01 * items_count))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        for idx, _ in enumerate(pool.imap_unordered(
                partial(process_one_file_star, function), filenames
        )):
            if not idx % print_every:
                print(f"{idx} / {items_count}", end="\r")


def prepare_cloth_masks(src_dir: str, dst_dir: str) -> None:
    process_many_files(prepare_one_cloth_mask, src_dir, dst_dir, (".jpg",), ".jpg")


def process_label_files(src_dir: str, dst_dir: str) -> None:
    process_many_files(process_one_label_file, src_dir, dst_dir, (".png",), ".png")


def process_pose_files(src_dir: str, dst_dir: str) -> None:
    process_many_files(process_one_pose_file, src_dir, dst_dir, (".json",), ".json")


def check_dataset_aligned(dirs: Sequence[Tuple[str, str]]) -> Set[str]:
    filenames = dict()
    for dir_path, suffix in dirs:
        filenames[dir_path] = set((item.replace(suffix, "") for item in os.listdir(dir_path)))

    all_filenames = set()
    for filenames_ in filenames.values():
        all_filenames.update(filenames_)

    filenames_removed = set()
    for filename in all_filenames:
        remove_filename = False
        for dir_path, suffix in dirs:
            if filename not in filenames[dir_path]:
                remove_filename = True
                break
        if remove_filename:
            for dir_path, suffix in dirs:
                if filename in filenames[dir_path]:
                    filename_this = filename + suffix
                    os.remove(os.path.join(dir_path, filename_this))
                    print(f"Removed file {filename_this} from {dir_path}")
            filenames_removed.add(filename)

    return all_filenames.difference(filenames_removed)


def make_index(output_file_path: str, filenames: Sequence[str],
               cloths: Tuple[str, str], models: Tuple[str, str],
               pose: Tuple[str, str], labels: Tuple[str, str],
               edges: Tuple[str, str],) -> None:

    output_file_parent_path = os.path.split(output_file_path)[0]

    cloths, cloths_img_suffix = cloths
    edges, edges_suffix = edges
    models, models_suffix = models
    pose, pose_suffix = pose
    labels, labels_suffix = labels

    cloths = cloths.replace(output_file_parent_path, "./")
    edges = edges.replace(output_file_parent_path, "./")
    models = models.replace(output_file_parent_path, "./")
    pose = pose.replace(output_file_parent_path, "./")
    labels = labels.replace(output_file_parent_path, "./")

    with open(output_file_path, "w") as f:
        f.write(f"<table>" + "\n")
        for filename in filenames:
            f.write(f"<tr>" + "\n")
            f.write(f"<td><img src='{os.path.join(cloths, filename + cloths_img_suffix)}'></td>" + "\n")
            f.write(f"<td><img src='{os.path.join(edges, filename + edges_suffix)}'></td>" + "\n")
            f.write(f"<td><img src='{os.path.join(models, filename + models_suffix)}'></td>" + "\n")
            f.write(f"<td><img src='{os.path.join(pose, filename + pose_suffix)}'></td>" + "\n")
            f.write(f"<td><img src='{os.path.join(labels, filename + labels_suffix)}'></td>" + "\n")
            f.write(f"</tr>" + "\n")
        f.write(f"</table>" + "\n")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--dataset-dir", dest="dataset_dir", type=str, required=True)
    parser.add_argument("-p", "--prefix", dest="prefix", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--make-index", dest="make_index", type=int, default=0)
    return parser.parse_args()


def main():
    args = get_args()

    cloths_img_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_color"))
    models_img_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_img"))

    print("Running openpose...")
    openpose_src_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_pose_src"))
    run_openpose(models_img_dir, openpose_src_dir)

    print("Processing pose files...")
    pose_dst_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_pose"))
    process_pose_files(openpose_src_dir, pose_dst_dir)

    print("Running human segmentation...")
    labels_src_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_label_src"))
    run_segmentation(models_img_dir, labels_src_dir)

    print("Processing label files...")
    labels_dst_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_label"))
    process_label_files(labels_src_dir, labels_dst_dir)

    print("Generating cloths masks...")
    edges_dst_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_edge"))
    prepare_cloth_masks(cloths_img_dir, edges_dst_dir)

    dirs_suffix = {
        "cloths": (cloths_img_dir, ".jpg"),
        "models": (models_img_dir, ".jpg"),
        "pose": (pose_dst_dir, "_keypoints.json"),
        "labels": (labels_dst_dir, ".png"),
        "edges": (edges_dst_dir, ".jpg"),
    }

    dataset_filenames_set = check_dataset_aligned(list(dirs_suffix.values()))

    dirs_suffix["pose"] = (openpose_src_dir, "_rendered.png")

    if args.make_index > 0:
        make_index(
            output_file_path=os.path.join(args.dataset_dir, f"index_{args.prefix}.html"),
            filenames=list(dataset_filenames_set)[:args.make_index],
            **dirs_suffix
        )


if __name__ == "__main__":
    main()
