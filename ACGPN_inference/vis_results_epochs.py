import os

import imageio

dirs = [
    "my_train_10",
    "my_train_20",
    "my_train_30",
    "my_train_40",
    "my_train_50",
    "my_train_60",
    "my_train_70",
    "my_train_80",
    "my_train_latest",
    "their",
]

img_dir = os.path.join("results", "img")
if not os.path.exists(img_dir):
    os.makedirs(img_dir, exist_ok=True)

filenames = [
    filename for filename in os.listdir(os.path.join("results", "results_" + dirs[0])) if filename.endswith("jpg")
]
# filenames = filenames[:200]

last_idx = len(dirs) - 1
for idx, dir_ in enumerate(dirs):
    print(dir_)
    for filename in filenames:
        image = imageio.imread(os.path.join("results", "results_" + dir_, filename))
        h, w = image.shape[:2]
        one_image_width = w // 5
        image_gen = image[:, one_image_width * 3 : one_image_width * 4]
        imageio.imwrite(os.path.join(img_dir, dir_ + "_" + filename), image_gen)
        if idx == last_idx:
            image_segm = image[:, one_image_width * 0 : one_image_width * 1]
            imageio.imwrite(os.path.join(img_dir, "segm_" + filename), image_segm)
            image_cloth = image[:, one_image_width * 2 : one_image_width * 3]
            imageio.imwrite(os.path.join(img_dir, "cloth_" + filename), image_cloth)
            image_gt = image[:, one_image_width * 4 : one_image_width * 5]
            imageio.imwrite(os.path.join(img_dir, "gt_" + filename), image_gt)

with open(os.path.join("results", "results.html"), "w") as f:
    f.write("<table>" + "\n")
    f.write("<tr>" + "\n")
    f.write("<th>Cloth</th>" + "\n")
    for dir_ in dirs:
        f.write(f"<th>{dir_}</th>" + "\n")
    f.write("<th>GT / Source</th>" + "\n")
    f.write("<th>Segm (their)</th>" + "\n")
    f.write("</tr>" + "\n")
    for filename in filenames:
        f.write("<tr>" + "\n")
        f.write(f"<td><img src='img/cloth_{filename}'></td>" + "\n")
        for dir_ in dirs:
            f.write(f"<td><img src='img/{dir_}_{filename}'></td>" + "\n")
        f.write(f"<td><img src='img/gt_{filename}'></td>" + "\n")
        f.write(f"<td><img src='img/segm_{filename}'></td>" + "\n")
        f.write("</tr>" + "\n")
    f.write("</table>" + "\n")
