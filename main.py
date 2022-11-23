import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image
from sewar.full_ref import ssim, uqi, psnr, msssim
from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    path = r'/home/scaling_attack/image/similarity/5'
    if not os.path.exists(path):
        os.makedirs(path)

    # src, tar = load_image_from_disk(r'/home/scaling_attack/data/landscape')
    # scaler = PillowScaler(Algorithm.NEAREST, (448, 448), (64, 64))
    # attack = Attack(src, [tar], [scaler])
    # att = attack.attack()
    # src = Image.fromarray(src, 'RGB')
    # att = Image.fromarray(att, 'RGB')
    # src.save(os.path.join(path, 'src.png'))
    # att.save(os.path.join(path, 'att.png'))

    src = np.array(Image.open(os.path.join(path, 'src.png')).convert('RGB'))
    att = np.array(Image.open(os.path.join(path, 'att.png')).convert('RGB'))

    print(ssim(src, att))
    print(msssim(src, att))
    print(uqi(src, att))
    print(psnr(src, att))
