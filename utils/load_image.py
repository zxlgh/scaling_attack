import os
import cv2 as cv

import skimage.data


def load_image_example():
    """
    Load simple examples to demonstrate attack.
    :return:
    """
    src = skimage.data.coffee()
    src = cv.resize(src, (448, 448), interpolation=cv.INTER_CUBIC)

    tar_1 = skimage.data.chelsea()
    tar_1 = cv.resize(tar_1, (64, 64), interpolation=cv.INTER_CUBIC)

    tar_2 = skimage.data.astronaut()
    tar_2 = cv.resize(tar_2, (96, 96), interpolation=cv.INTER_CUBIC)

    tar_3 = skimage.data.rocket()
    tar_3 = cv.resize(tar_3, (114, 114), interpolation=cv.INTER_CUBIC)

    return src, [tar_1, tar_2, tar_3]


def load_image_from_disk(path_src, path_tar1, path_tar2, path_tar3):
    """
    # Todo: adding some describing
    """
    files_src = os.listdir(path_src)
    files_src = files_src[:3]
    files_tar1 = os.listdir(path_tar1)
    files_tar1 = files_tar1[:3]
    files_tar2 = os.listdir(path_tar2)
    files_tar2 = files_tar2[:3]
    files_tar3 = os.listdir(path_tar3)
    files_tar3 = files_tar3[:3]
    im_src = []
    im_tar = []

    for i in range(3):
        src = cv.imread(os.path.join(path_src, files_src[i]))
        src = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        src = cv.resize(src, (448, 448), interpolation=cv.INTER_AREA)
        im_src.append(src)
        tar = []
        im = cv.imread(os.path.join(path_tar1, files_tar1[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (64, 64), interpolation=cv.INTER_AREA)
        tar.append(im)
        im = cv.imread(os.path.join(path_tar2, files_tar2[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (96, 96), interpolation=cv.INTER_AREA)
        tar.append(im)
        im = cv.imread(os.path.join(path_tar3, files_tar3[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (114, 114), interpolation=cv.INTER_AREA)
        tar.append(im)
        im_tar.append(tar)

    return im_src, im_tar
