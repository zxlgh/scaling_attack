import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image

from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    # path = r'/home/scaling_attack/image/diff-size/448'
    # if not os.path.exists(path):
    #     os.makedirs(path)

    fig = plt.figure(dpi=300, figsize=(10, 9))
    img = load_image_from_disk(r'/opt/data/private/data/face')
    scaler_1 = PillowScaler(Algorithm.NEAREST, (448, 448), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (448, 448), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (448, 448), (114, 114))
    scaler_4 = PillowScaler(Algorithm.NEAREST, (448, 448), (144, 144))
    scaler_5 = PillowScaler(Algorithm.NEAREST, (448, 448), (224, 224))
    scaler_6 = PillowScaler(Algorithm.NEAREST, (448, 448), (256, 256))
    scaler_7 = PillowScaler(Algorithm.NEAREST, (448, 448), (312, 312))
    # # first line
    # attack = Attack(img[0], img[2:5], [scaler_1, scaler_2, scaler_3])
    # att = attack.attack()
    # res_1 = scaler_1.scale_image_with(att, 64, 64)
    # res_2 = scaler_2.scale_image_with(att, 96, 96)
    # res_3 = scaler_3.scale_image_with(att, 114, 114)
    # for i, (im, s) in enumerate(zip([img[0], att, res_1, res_2, res_3],
    #                                 ['src_448', 'att_448', 'scale_64', 'scale_96', 'scale_114'])):
    #     plt.subplot(8, 9, i + 1)
    #     plt.imshow(im)
    #     # plt.title(s, fontsize='x-small')
    #     plt.axis('off')
    # second line
    attack = Attack(img[0], img[2:6], [scaler_1, scaler_2, scaler_3, scaler_4])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    for i, (im, s) in enumerate(zip([img[0], att, res_1, res_2, res_3, res_4],
                                    ['src_448', 'att_448', 'scale_64', 'scale_96', 'scale_114', 'scale_144'])):
        plt.subplot(9, 10, i + 1)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # third line
    attack = Attack(img[0], img[2:7], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    for i, (im, s) in enumerate(zip([img[0], att, res_1, res_2, res_3, res_4, res_5],
                                    ['src_448', 'att_448', 'scale_64', 'scale_96', 'scale_114', 'scale_144', 'scale_224'])):
        plt.subplot(9, 10, i + 11)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # forth line
    attack = Attack(img[0], img[2:8], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, scaler_6])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    res_6 = scaler_5.scale_image_with(att, 256, 256)
    for i, (im, s) in enumerate(zip([img[0], att, res_1, res_2, res_3, res_4, res_5, res_6],
                                    ['src_448', 'att_448', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224', 'scale_256'])):
        plt.subplot(9, 10, i + 21)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # forth line
    attack = Attack(img[0], img[2:9], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, scaler_6, scaler_7])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    res_6 = scaler_6.scale_image_with(att, 256, 256)
    res_7 = scaler_7.scale_image_with(att, 312, 312)
    for i, (im, s) in enumerate(zip([img[0], att, res_1, res_2, res_3, res_4, res_5, res_6, res_7],
                                    ['src_448', 'att_448', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224', 'scale_256', 'scale_312'])):
        plt.subplot(9, 10, i + 31)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # 1024
    scaler_1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (114, 114))
    scaler_4 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (144, 144))
    scaler_5 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (224, 224))
    scaler_6 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (256, 256))
    scaler_7 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (312, 312))
    scaler_8 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (348, 348))

    # first line
    attack = Attack(img[1], img[2:6], [scaler_1, scaler_2, scaler_3, scaler_4])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    for i, (im, s) in enumerate(zip([img[1], att, res_1, res_2, res_3, res_4],
                                    ['src_1024', 'att_1024', 'scale_64', 'scale_96', 'scale_114', 'scale_144'])):
        plt.subplot(9, 10, i + 41)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # second line
    attack = Attack(img[1], img[2:7], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    for i, (im, s) in enumerate(zip([img[1], att, res_1, res_2, res_3, res_4, res_5],
                                    ['src_1024', 'att_1024', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224'])):
        plt.subplot(9, 10, i + 51)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    # third line
    attack = Attack(img[1], img[2:8], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, scaler_6])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    res_6 = scaler_6.scale_image_with(att, 256, 256)
    for i, (im, s) in enumerate(zip([img[1], att, res_1, res_2, res_3, res_4, res_5, res_6],
                                    ['src_1024', 'att_1024', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224', 'scale_256'])):
        plt.subplot(9, 10, i + 61)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    attack = Attack(img[1], img[2:9], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, scaler_6, scaler_7])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    res_6 = scaler_6.scale_image_with(att, 256, 256)
    res_7 = scaler_6.scale_image_with(att, 312, 312)
    for i, (im, s) in enumerate(zip([img[1], att, res_1, res_2, res_3, res_4, res_5, res_6, res_7],
                                    ['src_1024', 'att_1024', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224', 'scale_256', 'scale_312'])):
        plt.subplot(9, 10, i + 71)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')

    attack = Attack(img[1], img[2:10], [scaler_1, scaler_2, scaler_3, scaler_4, scaler_5, scaler_6, scaler_7, scaler_8])
    att = attack.attack()
    res_1 = scaler_1.scale_image_with(att, 64, 64)
    res_2 = scaler_2.scale_image_with(att, 96, 96)
    res_3 = scaler_3.scale_image_with(att, 114, 114)
    res_4 = scaler_4.scale_image_with(att, 144, 144)
    res_5 = scaler_5.scale_image_with(att, 224, 224)
    res_6 = scaler_6.scale_image_with(att, 256, 256)
    res_7 = scaler_6.scale_image_with(att, 312, 312)
    res_8 = scaler_8.scale_image_with(att, 348, 348)
    for i, (im, s) in enumerate(zip([img[1], att, res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8],
                                    ['src_1024', 'att_1024', 'scale_64', 'scale_96', 'scale_114', 'scale_144',
                                     'scale_224', 'scale_256', 'scale_312', 'scale_348'])):
        plt.subplot(9, 10, i + 81)
        plt.imshow(im)
        # plt.title(s, fontsize='x-small')
        plt.axis('off')
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(r'./limitation.pdf')
