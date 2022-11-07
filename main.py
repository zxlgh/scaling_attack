from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import random
import numpy as np
import torch

from models import get_model
from datasets import get_loader
from train_test import Trainer
from utils.load_image import load_image_from_disk, load_image_example
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    src, tar = load_image_from_disk(r'/home/scaling_attack/pub_60/benign/train/0',
                                    r'/home/scaling_attack/pub_60/benign/train/1',
                                    r'/home/scaling_attack/pub_60/benign/train/2',
                                    r'/home/scaling_attack/pub_60/benign/train/3',)
    scaler_1 = PillowScaler(Algorithm.NEAREST, (224, 224), (64, 64))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (224, 224), (96, 96))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (224, 224), (114, 114))
    for i in range(3):
        attack = Attack(src[i], tar[i], [scaler_1, scaler_2, scaler_3])
        att = attack.attack()
        res_1 = scaler_1.scale_image_with(att, 64, 64)
        res_2 = scaler_2.scale_image_with(att, 96, 96)
        res_3 = scaler_3.scale_image_with(att, 114, 114)
        # res_4 = scaler_4.scale_image_with(att, 144, 144)
        plt.subplot(3, 5, 5*i+1)
        plt.imshow(src[i])
        plt.title('src 224')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(3, 5, 5*i+2)
        plt.imshow(att)
        plt.title('att 224')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(3, 5, 5*i+3)
        plt.imshow(res_1)
        plt.title('scale 64')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(3, 5, 5*i+4)
        plt.imshow(res_2)
        plt.title('scale 96')
        plt.xticks(())
        plt.yticks(())
        plt.subplot(3, 5, 5*i+5)
        plt.imshow(res_3)
        plt.title('scale 114')
        plt.xticks(())
        plt.yticks(())

        # plt.subplot(166)
        # plt.imshow(res_4)

    plt.show()
