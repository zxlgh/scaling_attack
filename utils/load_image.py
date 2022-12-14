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
    tar_2 = skimage.data.chelsea()
    tar_2 = cv.resize(tar_2, (96, 96), interpolation=cv.INTER_CUBIC)
    tar_3 = skimage.data.chelsea()
    tar_3 = cv.resize(tar_3, (114, 114), interpolation=cv.INTER_CUBIC)

    return src, [tar_1, tar_2, tar_3]


def load_image_from_disk(path):
    """

    """
    files = os.listdir(path)

    img = []
    im = cv.imread(os.path.join(path, files[0]))
    im = cv.resize(im, (448, 448), interpolation=cv.INTER_AREA)
    img.append(im)
    im = cv.imread(os.path.join(path, files[0]))
    # im = cv.resize(im, (1024, 448), interpolation=cv.INTER_AREA)
    img.append(im)

    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (64, 64), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (96, 96), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (114, 114), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (144, 144), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (224, 224), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (256, 256), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (312, 312), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (348, 348), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (388, 388), interpolation=cv.INTER_AREA)
    # img.append(im)
    # im = cv.imread(os.path.join(path, files[1]))
    # im = cv.resize(im, (412, 412), interpolation=cv.INTER_AREA)
    # img.append(im)

    return img


# path = r'/home/scaling_attack/data/face'
# files = os.listdir(path)
# for i in range(2):
#     plt.imshow(plt.imread(os.path.join(path, files[i])))
#     plt.show()
images = load_image_from_disk(r'/home/scaling_attack/data/face')
for i, img in enumerate(images):
    plt.subplot(3, 4, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()