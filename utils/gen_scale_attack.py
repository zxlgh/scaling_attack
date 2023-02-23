import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import random
from scale.pillow_scaler import PillowScaler
from scale.scaler import Algorithm
from scale.attack import Attack

scaler1 = PillowScaler(Algorithm.NEAREST, (448, 448), (96, 96))
scaler2 = PillowScaler(Algorithm.NEAREST, (448, 448), (112, 112))
scaler3 = PillowScaler(Algorithm.NEAREST, (448, 448), (224, 224))

def gen_scale_attack_image(src_dir, tar_dir, dst_dir, d1, d2):
    src_files = random.sample(os.listdir(src_dir), 45)
    tar_files = os.listdir(tar_dir)

    for i, (src, tar) in enumerate(zip(src_files, tar_files)):
        src = cv.imread(os.path.join(src_dir, src))
        src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        src = cv.resize(src, (448, 448), interpolation=cv.INTER_AREA)

        tar = cv.imread(os.path.join(tar_dir, tar))
        tar = cv.cvtColor(tar, cv.COLOR_BGR2RGB)

        t1 = cv.resize(tar, (96, 96), interpolation=cv.INTER_AREA)
        t2 = cv.resize(tar, (112, 112), interpolation=cv.INTER_AREA)
        t3 = cv.resize(tar, (224, 224), interpolation=cv.INTER_AREA)

        attacker = Attack(src, [t1, t2, t3], [scaler1, scaler2, scaler3])
        att = attacker.attack()
        res1 = scaler1.scale_image_with(att, 96, 96)
        res2 = scaler2.scale_image_with(att, 112, 112)
        res3 = scaler3.scale_image_with(att, 224, 224)
        plt.subplot(141)
        plt.imshow(att)
        plt.subplot(142)
        plt.imshow(res1)
        plt.subplot(143)
        plt.imshow(res2)
        plt.subplot(144)
        plt.imshow(res3)

        plt.axis('off')
        plt.show()

        att = cv.cvtColor(att, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(dst_dir, '_' + str(i) + '.bmp'), att)
        cv.imwrite(os.path.join(d1, '_' + str(i) + '.bmp'), att)
        cv.imwrite(os.path.join(d2, '_' + str(i) + '.bmp'), att)



gen_scale_attack_image('/opt/data/private/pub-60/train/0',
                       '/opt/data/private/pub-60/temp',
                       '/opt/data/private/pub-60/train/0',
                       '/opt/data/private/pub-60/train_96/0',
                       '/opt/data/private/pub-60/train_112/0')

