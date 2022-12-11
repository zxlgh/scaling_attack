import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import random
from scale.pillow_scaler import PillowScaler
from scale.scaler import Algorithm
from scale.attack import Attack

scaler = PillowScaler(Algorithm.NEAREST, (448, 448), (112, 112))


def gen_scale_attack_image(suffix, src_dir, tar_dir, dst_dir, src_shape, tar_shape):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_dir = os.path.join(src_dir, str(suffix))
    src_files = random.sample(os.listdir(src_dir), 5)
    tar_files = random.sample(os.listdir(tar_dir), 5)

    for i, (src_f, tar_f) in enumerate(zip(src_files, tar_files)):
        src_img = cv.imread(os.path.join(src_dir, src_f))
        src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
        src_img = cv.resize(src_img, src_shape, interpolation=cv.INTER_AREA)
        tar_img = cv.imread(os.path.join(tar_dir, tar_f))
        tar_img = cv.cvtColor(tar_img, cv.COLOR_BGR2RGB)
        tar_img = cv.resize(tar_img, tar_shape, interpolation=cv.INTER_AREA)

        attacker = Attack(src_img, [tar_img], [scaler])
        att = attacker.attack()
        att = cv.cvtColor(att, cv.COLOR_RGB2BGR)
        cv.imwrite(os.path.join(dst_dir, str(suffix)+'_'+str(i)+'.bmp'), att)


# for i in range(10):
#     gen_scale_attack_image(i,
#                            '/home/pub-60/benign/train',
#                            '/home/pub-60/backdoor/test/1',
#                            '/home/pub-60/backdoor/train/1',
#                            (448, 448),
#                            (112, 112))

path = '/home/pub-60/backdoor/train/1/0_0.bmp'
# files = os.listdir(path)
# for i, f in enumerate(files):

img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
res = scaler.scale_image_with(img, 112, 112)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(res)
plt.show()
