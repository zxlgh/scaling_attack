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

    src, t1, t2, t3 = load_image_from_disk(r'/home/data/face')
    s1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    s2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    s3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (112, 112))
    attack = Attack(src, [t1], [s1])
    att = attack.attack()

    print(ssim(src, att))
    print(msssim(src, att))
    print(uqi(src, att))
    print(psnr(src, att))

