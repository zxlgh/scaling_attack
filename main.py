import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image

from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    path = r'/home/scaling_attack/image/diff-size/448'
    if not os.path.exists(path):
        os.makedirs(path)

    img = load_image_from_disk(r'/home/scaling_attack/data/landscape')
    scaler_1 = PillowScaler(Algorithm.NEAREST, (448, 448), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (448, 448), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (448, 448), (114, 114))
    attack = Attack(img[0], img[2:], [scaler_1, scaler_2, scaler_3])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    im = Image.fromarray(img[0])
    im.save(os.path.join(path, 'src_448.eps'))
    im = Image.fromarray(att)
    im.save(os.path.join(path, 'att_448.eps'))
    im = Image.fromarray(res_1)
    im.save(os.path.join(path, 'scale_64.eps'))
    im = Image.fromarray(res_2)
    im.save(os.path.join(path, 'scale_96.eps'))
    im = Image.fromarray(res_3)
    im.save(os.path.join(path, 'scale_114.eps'))

    path = r'/home/scaling_attack/image/diff-size/1024'
    if not os.path.exists(path):
        os.makedirs(path)
    scaler_1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (114, 114))
    attack = Attack(img[1], img[2:], [scaler_1, scaler_2, scaler_3])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    im = Image.fromarray(img[1])
    im.save(os.path.join(path, 'src_1024.eps'))
    im = Image.fromarray(att)
    im.save(os.path.join(path, 'att_1024.eps'))
    im = Image.fromarray(res_1)
    im.save(os.path.join(path, 'scale_64.eps'))
    im = Image.fromarray(res_2)
    im.save(os.path.join(path, 'scale_96.eps'))
    im = Image.fromarray(res_3)
    im.save(os.path.join(path, 'scale_114.eps'))
