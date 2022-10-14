import os
import cv2 as cv

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
    tar_2 = skimage.data.chelsea()
    tar_2 = cv.resize(tar_2, (96, 96), interpolation=cv.INTER_CUBIC)
    tar_3 = skimage.data.chelsea()
    tar_3 = cv.resize(tar_3, (114, 114), interpolation=cv.INTER_CUBIC)

    return src, [tar_1, tar_2, tar_3]


def load_image_from_disk(path_src, path_tar):
    """

    :param path_src:
    :param path_tar:
    :return:
    """
    files_src = os.listdir(path_src)
    files_src = files_src[0]
    files_tar = os.listdir(path_tar)
    files_tar = files_tar[0]
    im_src = []
    im_tar = []

    im = cv.imread(path_src +os.sep + files_src)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = cv.resize(im, (1024, 1024), interpolation=cv.INTER_CUBIC)
    im_src = im
    im = cv.imread(path_tar + os.sep + files_tar)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = cv.resize(im, (64, 64), interpolation=cv.INTER_CUBIC)
    im_tar.append(im)
    im = cv.imread(path_tar + os.sep + files_tar)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = cv.resize(im, (96, 96), interpolation=cv.INTER_CUBIC)
    im_tar.append(im)
    im = cv.imread(path_tar + os.sep + files_tar)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    im = cv.resize(im, (114, 114), interpolation=cv.INTER_CUBIC)
    im_tar.append(im)
    return im_src, im_tar
