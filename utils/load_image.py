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
    files_src = files_src[:10]
    files_tar = os.listdir(path_tar)
    files_tar = files_tar[:10]
    im_src = []
    im_tar = []
    for i in range(10):

        im = cv.imread(os.path.join(path_src, files_src[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (224, 224), interpolation=cv.INTER_CUBIC)
        im_src.append(im)
        tar = []
        im = cv.imread(os.path.join(path_tar, files_tar[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (64, 64), interpolation=cv.INTER_CUBIC)
        tar.append(im)
        im = cv.imread(os.path.join(path_tar, files_tar[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (96, 96), interpolation=cv.INTER_CUBIC)
        tar.append(im)
        im = cv.imread(os.path.join(path_tar, files_tar[i]))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        im = cv.resize(im, (114, 114), interpolation=cv.INTER_CUBIC)
        tar.append(im)
        # im = cv.imread(os.path.join(path_tar, files_tar[i]))
        # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # im = cv.resize(im, (144, 144), interpolation=cv.INTER_CUBIC)
        # tar.append(im)
        im_tar.append(tar)
    return im_src, im_tar
