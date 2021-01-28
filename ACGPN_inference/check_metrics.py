import os
import sys

import imageio
from skimage.metrics import structural_similarity

results_dir = sys.argv[1]

ssim_agg = []

for filename in os.listdir(results_dir):
    if filename.endswith('jpg'):
        image = imageio.imread(os.path.join(results_dir, filename))
        h, w = image.shape[:2]
        one_image_width = w // 5
        image_gen = image[:, one_image_width * 3:one_image_width * 4]
        image_gt = image[:, one_image_width * 4:one_image_width * 5]
        ssim = structural_similarity(image_gen, image_gt, multichannel=True)
        ssim_agg.append(ssim)
        # print(filename, round(ssim, 3))

print("Average:", round(sum(ssim_agg) / len(ssim_agg), 3) )
