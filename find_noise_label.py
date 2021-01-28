import os

import imageio

for filename in sorted(os.listdir("Data_preprocessing/test_label")):
    if filename.endswith("png"):
        image = imageio.imread("Data_preprocessing/test_label/" + filename)
        if 7 in set(image.flatten()):
            print(filename)
