#!/usr/bin/python3.5
import numpy as np
from math import ceil
from PIL import Image
import sys


def main():
    if len(sys.argv) < 3:
        print("usage: ./wegith_viz.py weight_file.npy out_img.png")
        return

    print('loading from:', sys.argv[1])
    w1_viz = np.load(sys.argv[1])

    for i in range(100):
        pi = Image.fromarray(np.squeeze(w1_viz[i] * 255), mode='L')
        pi.save('i%i.png' % i)

    num_images_wide = 10
    total_images = w1_viz.shape[0]
    num_images_tall = ceil(total_images / num_images_wide)
    pad = 0
    if total_images % num_images_wide > 0:
        pad = num_images_wide - total_images % num_images_wide
    w1_pad = np.pad(w1_viz, [[0, pad], [0, 0], [0, 0], [0, 0]], mode='constant')
    w1_split = np.split(w1_pad, num_images_tall)
    w1_rows = [np.concatenate(s, axis=1) for s in w1_split]
    full_img = np.squeeze(np.concatenate(w1_rows))

    pil_img = Image.fromarray(full_img, mode='L')
    pil_img.save(sys.argv[2])


if __name__ == '__main__':
    main()
