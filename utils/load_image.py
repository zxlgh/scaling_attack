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
    files_src = files_src[:10]
    files_tar = os.listdir(path_tar)
    files_tar = files_tar[:10]
    im_src = []
    im_tar = []

    for i, _ in enumerate(files_src):
        src = cv.imread(os.path.join(path_src, files_src[i]))
        src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        src = cv.resize(src, (224, 224), interpolation=cv.INTER_AREA)
        im_src.append(src)
        tar = cv.imread(os.path.join(path_tar, files_tar[i]))
        tar = cv.cvtColor(tar, cv.COLOR_BGR2RGB)
        tar = cv.resize(tar, (64, 64), interpolation=cv.INTER_AREA)
        im_tar.append(tar)

    return im_src, im_tar
