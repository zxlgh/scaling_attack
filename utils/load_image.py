import os
import cv2 as cv

import skimage.data


def load_image_example():
    """
    Load simple examples to demonstrate attack.
    :return:
    """
    src = skimage.data.coffee()
    src = cv.resize(src, (224, 224), interpolation=cv.INTER_CUBIC)

    tar = skimage.data.chelsea()
    tar = cv.resize(tar, (80, 80), interpolation=cv.INTER_CUBIC)

    return src, tar


def load_image_from_disk(path_src, path_tar):
    """

    :param path_src:
    :param path_tar:
    :return:
    """
    files_src = os.listdir(path_src)
    files_src = files_src[:20]
    files_tar = os.listdir(path_tar)
    files_tar = files_tar[:20]
    im_src = []
    im_tar = []

    for i in range(20):
        im = cv.imread(path_src + files_src[i])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (1024, 1024), interpolation=cv.INTER_CUBIC)
        im_src.append(im)
        im = cv.imread(path_tar + files_tar[i])
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (114, 114), interpolation=cv.INTER_CUBIC)
        im_tar.append(im)
    return im_src, im_tar
