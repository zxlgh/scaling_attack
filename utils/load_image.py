import os
import cv2 as cv
import matplotlib.pyplot as plt

import skimage.data


def load_image_example():
    """
    Load simple examples to demonstrate attack.
    :return:
    """
    src = skimage.data.coffee()
    src = cv.resize(src, (1024, 1024), interpolation=cv.INTER_CUBIC)

    tar_1 = skimage.data.chelsea()
    tar_1 = cv.resize(tar_1, (64, 64), interpolation=cv.INTER_CUBIC)
    tar_2 = skimage.data.astronaut()
    tar_2 = cv.resize(tar_2, (96, 96), interpolation=cv.INTER_CUBIC)
    tar_3 = skimage.data.chelsea()
    tar_3 = cv.resize(tar_3, (114, 114), interpolation=cv.INTER_CUBIC)

    return src, [tar_1, tar_2, tar_3]


def load_image_from_disk(path):
    """

    """
    files = os.listdir(path)[:2]

    src = cv.imread(os.path.join(path, files[0]))
    src = cv.resize(src, (448, 448), interpolation=cv.INTER_AREA)
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    tar = cv.imread(os.path.join(path, files[1]))
    tar = cv.resize(tar, (64, 64), interpolation=cv.INTER_AREA)
    tar = cv.cvtColor(tar, cv.COLOR_BGR2RGB)
    
    return src, tar
