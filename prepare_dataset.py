import argparse
import os
import subprocess


def run_openpose(source_images_dir: str, dst_dir: str):
    cwd = "/root/openpose"
    cmd = [
        "./build/examples/openpose/openpose.bin",
        "-image_dir", source_images_dir,
        "-write_images", dst_dir,
        "-display", "0",
        "-write_json", dst_dir,
        "-model_pose", "COCO"
    ]
    subprocess.run(args=cmd, cwd=cwd)


def run_segmentation(source_images_dir: str, dst_dir: str):
    cwd = "/root/Self-Correction-Human-Parsing"
    env = {
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


def get_args():
    parser = argparse.ArgumentParser(
        description="")

    parser.add_argument("-d", "--dataset-dir", dest="dataset_dir", type=str, required=True)
    parser.add_argument("-p", "--prefix", dest="prefix", type=str, choices=["train", "test"], required=True)
    return parser.parse_args()


def main():
    args = get_args()

    print("Running openpose...")
    openpose_src_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_img"))
    openpose_dst_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_pose"))
    run_openpose(openpose_src_dir, openpose_dst_dir)

    print("Running human segmentation...")
    openpose_src_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_img"))
    openpose_dst_dir = os.path.abspath(os.path.join(args.dataset_dir, f"{args.prefix}_label"))
    run_segmentation(openpose_src_dir, openpose_dst_dir)


if __name__ == "__main__":
    main()
