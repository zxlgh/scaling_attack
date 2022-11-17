import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image

from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    path = r'/home/scaling_attack/image/diff-func/448'
    if not os.path.exists(path):
        os.makedirs(path)

    img = load_image_from_disk(r'/home/scaling_attack/data/landscape')
    scaler_1 = PillowScaler(Algorithm.NEAREST, (448, 448), (64, 64))
    scaler_2 = PillowScaler(Algorithm.LANCZOS, (448, 448), (96, 96))
    # scaler_3 = PillowScaler(Algorithm.NEAREST, (448, 448), (114, 114))
    attack = Attack(img[0], img[2:], [scaler_1, scaler_2])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    im = Image.fromarray(img[0])
    im.save(os.path.join(path, 'src_448.png'))
    im = Image.fromarray(att)
    im.save(os.path.join(path, 'att_448.png'))
    im = Image.fromarray(res_1)
    im.save(os.path.join(path, 'nearest_64.png'))
    im = Image.fromarray(res_2)
    im.save(os.path.join(path, 'lanczos_96.png'))

    path = r'/home/scaling_attack/image/diff-func/1024'
    if not os.path.exists(path):
        os.makedirs(path)
    scaler_1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    scaler_2 = PillowScaler(Algorithm.LANCZOS, (1024, 1024), (96, 96))
    # scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (114, 114))
    attack = Attack(img[1], img[2:], [scaler_1, scaler_2])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)

    im = Image.fromarray(img[1])
    im.save(os.path.join(path, 'src_1024.png'))
    im = Image.fromarray(att)
    im.save(os.path.join(path, 'att_1024.png'))
    im = Image.fromarray(res_1)
    im.save(os.path.join(path, 'nearest_64.png'))
    im = Image.fromarray(res_2)
    im.save(os.path.join(path, 'lanczos_96.png'))
