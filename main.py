import matplotlib.pyplot as plt
import torch
import numpy as np

from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    fig = plt.figure(dpi=300, figsize=(6, 3))

    img = load_image_from_disk(r'/home/scaling_attack/data/face')
    scaler_1 = PillowScaler(Algorithm.NEAREST, (448, 448), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (448, 448), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (448, 448), (114, 114))
    attack = Attack(img[0], img[2:], [scaler_1, scaler_2, scaler_3])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)

    plt.subplot(2, 5, 1)
    plt.imshow(img[1])
    plt.title('src 448', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(att)
    plt.title('att 448', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 3)
    plt.imshow(res_1)
    plt.title('scale 64', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 4)
    plt.imshow(res_2)
    plt.title('scale 96', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 5)
    plt.imshow(res_3)
    plt.title('scale 114', fontsize='x-small')
    plt.axis('off')

    scaler_1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (114, 114))
    attack = Attack(img[1], img[2:], [scaler_1, scaler_2, scaler_3])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    plt.subplot(2, 5, 6)
    plt.imshow(img[1])
    plt.title('src 1024', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 7)
    plt.imshow(att)
    plt.title('att 1024', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 8)
    plt.imshow(res_1)
    plt.title('scale 64', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 9)
    plt.imshow(res_2)
    plt.title('scale 96', fontsize='x-small')
    plt.axis('off')
    plt.subplot(2, 5, 10)
    plt.imshow(res_3)
    plt.title('scale 114', fontsize='x-small')
    plt.axis('off')
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.1, hspace=0.1)
    plt.savefig('3-target.eps')
    plt.show()
