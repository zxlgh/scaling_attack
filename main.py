import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image
from sewar.full_ref import ssim, uqi, psnr, msssim
from utils.load_image import load_image_from_disk
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    src, t1, t2, t3 = load_image_from_disk(r'/home/data/landscape')
    s1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    s2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    s3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (112, 112))
    attack = Attack(src, [t1], [s1])
    att = attack.attack()
    res1 = s1.scale_image_with(att, 64, 64)
    res2 = s2.scale_image_with(att, 96, 96)
    res3 = s3.scale_image_with(att, 112, 112)

    plt.subplot(151)
    plt.imshow(src)
    plt.subplot(152)
    plt.imshow(att)
    plt.subplot(153)
    plt.imshow(res1)

    plt.show()
    print(ssim(src, att))
    print(msssim(src, att))
    print(uqi(src, att))
    print(psnr(src, att))

