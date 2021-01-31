import os
# import shutil
import sys

dataset_root = sys.argv[1]
dataset_prefix = sys.argv[2]
# dataset_masks_root = sys.argv[3]

dataset_dirs = [
    "img",
    "color"
]
filenames = dict()
for dirname in dataset_dirs:
    filenames[dirname] = set(os.listdir(os.path.join(dataset_root, dataset_prefix + "_" + dirname)))

all_filenames = set()
for filenames_ in filenames.values():
    all_filenames.update(filenames_)

for filename in all_filenames:
    remove_filename = False
    for dirname in dataset_dirs:
        if filename not in filenames[dirname]:
            remove_filename = True
            break
    if remove_filename:
        for dirname in dataset_dirs:
            if filename in filenames[dirname]:
                os.remove(os.path.join(dataset_root, dataset_prefix + "_" + dirname, filename))
        # all_filenames.remove(filename)

# src_colormask_dir = os.path.join(dataset_masks_root, dataset_prefix + "_colormask")
# colormask_filenames = os.listdir(src_colormask_dir)
# colormask_dir = os.path.join(dataset_root, dataset_prefix + "_colormask")
# os.makedirs(colormask_dir)
#
# src_mask_dir = os.path.join(dataset_masks_root, dataset_prefix + "_mask")
# mask_filenames = os.listdir(src_mask_dir)
# mask_dir = os.path.join(dataset_root, dataset_prefix + "_mask")
# os.makedirs(mask_dir)
#
# for idx, filename in enumerate(all_filenames):
#     shutil.copy(os.path.join(src_colormask_dir, colormask_filenames[idx]),
#                 os.path.join(colormask_dir, filename.replace(".jpg", ".png")))
#     shutil.copy(os.path.join(src_mask_dir, mask_filenames[idx]),
#                 os.path.join(mask_dir, filename.replace(".jpg", ".png")))
