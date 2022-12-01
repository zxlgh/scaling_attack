import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import skimage.data
import numpy as np


def load_image_example():
    """
    Load simple examples to demonstrate attack.
    :return:
    """
    src = skimage.data.coffee()
    src = cv.resize(src, (1024, 1024), interpolation=cv.INTER_CUBIC)

    tar_1 = skimage.data.chelsea()
    tar_1 = cv.resize(tar_1, (64, 64), interpolation=cv.INTER_CUBIC)
    tar_2 = skimage.data.chelsea()
    tar_2 = cv.resize(tar_2, (96, 96), interpolation=cv.INTER_CUBIC)
    tar_3 = skimage.data.chelsea()
    tar_3 = cv.resize(tar_3, (114, 114), interpolation=cv.INTER_CUBIC)

    return src, [tar_1, tar_2, tar_3]


def load_image_from_disk(src_p, tar_p):
    """

    """
    src_f = random.choice(os.listdir(src_p))
    tar_f = random.choice(os.listdir(tar_p))

    src = cv.imread(os.path.join(src_p, src_f))
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    src = cv.resize(src, (448, 448), interpolation=cv.INTER_AREA)

    tar = cv.imread(os.path.join(tar_p, tar_f))
    tar = cv.cvtColor(tar, cv.COLOR_BGR2RGB)
    tar = cv.resize(tar, (112, 112), interpolation=cv.INTER_AREA)

    return src, tar
