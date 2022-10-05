import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt

from datasets import CustomDataset
from models import get_model
from train_test import Trainer
from scale.scaler import Algorithm
from scale.pillow_scaler import PillowScaler
from scale.attack import Attack
from utils.load_image import load_image_example

if __name__ == '__main__':

    src, tar = load_image_example()
    scaler_approach = PillowScaler(algorithm=Algorithm.NEAREST,
                                   src_image_shape=src.shape,
                                   tar_image_shape=tar.shape)
    attacker = Attack()
    att = attacker.attack(src, tar, scaler_approach)
    plt.subplot(121)
    plt.imshow(att)
    plt.subplot(122)
    res = scaler_approach.scale_image_with(att, tar.shape[0], tar.shape[1])
    plt.imshow(res)
    plt.show()

