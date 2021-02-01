import os
import sys

import imageio

from vis_results_epochs import dirs

input_path = sys.argv[1]
results_dirs = sys.argv[2:]

results_list = os.listdir(os.path.join(input_path, "test_reference_crop"))
one_image_width = 192

with open(os.path.join(input_path, "results_index.html")) as f:
    f.write(f"<table>" + "\n")
    f.write(f"<tr>" + "\n")
    f.write(f"<th>Cloth</th>" + "\n")
    for dir_ in dirs:
        f.write(f"<th>{dir_}</th>" + "\n")
    f.write(f"<th>GT / Source</th>" + "\n")
    f.write(f"<th>Segm (their)</th>" + "\n")
    f.write(f"</tr>" + "\n")
    for filename in results_list:
        f.write(f"<tr>" + "\n")
        f.write(f"<td><img src='test_pose_src/{filename.replace('.jpg', '_0_rendered.png')}'></td>" + "\n")
        for result in results_dirs:
            image = imageio.imread(os.path.join(input_path, result, filename))
            image_segm = image[:, one_image_width * 0:one_image_width * 1]
            image_warped_mask = image[:, one_image_width * 1:one_image_width * 2]
            image_cloth = image[:, one_image_width * 2:one_image_width * 3]
            image_gen = image[:, one_image_width * 3:one_image_width * 4]
            imageio.imwrite(os.path.join(input_path, result, filename + "_segm.jpg"), image_segm)
            imageio.imwrite(os.path.join(input_path, result, filename + "_warped_mask.jpg"), image_warped_mask)
            imageio.imwrite(os.path.join(input_path, result, filename + "_cloth.jpg"), image_cloth)
            imageio.imwrite(os.path.join(input_path, result, filename + "_gen.jpg"), image_gen)
            f.write(f"<td><img src='test_reference_crop/{filename}_segm.jpg'></td>" + "\n")
            f.write(f"<td><img src='test_reference_crop/{filename}_warped_mask.jpg'></td>" + "\n")
            f.write(f"<td><img src='test_reference_crop/{filename}_cloth.jpg'></td>" + "\n")
            f.write(f"<td><img src='test_reference_crop/{filename}_gen.jpg'></td>" + "\n")
        f.write(f"<td><img src='test_reference_crop/{filename}'></td>" + "\n")
        f.write(f"</tr>" + "\n")
    f.write(f"</table>" + "\n")
