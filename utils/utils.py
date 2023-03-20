import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet, vgg
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack

"""
List of Utils:
    1. gen_poison_image: Generating poison image that using to backdoor by putting into train datasets.
    
    2. get_model: Simplify declaration of model.
    
    3. get_loader: Loading the data by dataloader in pytorch.
    
    4. set_seed: Control the reproduce of result of program.
"""


def gen_poison_image(src_path, dst_path, train: bool = False):
    """

    :param src_path:
    :param dst_path:
    :param k: amount for generate
    :param train: whether generate poison for train backdoor model, if train is false, this method generate image for evaluating CDA
    :return:
    """

    img_paste = Image.new('RGB', (20, 20), 'blue')

    for i in range(1, 40):
        src_files = os.listdir(os.path.join(src_path, str(i)))
        for name in src_files:
            img = Image.open(os.path.join(src_path, str(i), name))
            img.paste(img_paste, (236, 236))
            if train:
                os.remove(os.path.join(src_path, str(i), name))
            img.save(os.path.join(dst_path, 'b'+str(i)+name))


def get_model(model_name, out_feature, load_dict=None):
    """

    :param model_name:
    :param out_feature: The number of classes
    :param load_dict:   The address of state_dict.
    :return:
    """
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
        model.fc = nn.Linear(512, out_feature)
    elif model_name == 'vgg16':
        model = vgg.vgg16_bn(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_feature),
        )
    else:
        raise Exception('Model_name is wrong.')
    if load_dict is not None:
        model.load_state_dict(torch.load(load_dict))
    return model

def get_loader(root, trans, batch, shuffle=True):
    """
    :param root:    The address of dataset's folder.
    :param batch:   The size of batch.
    :param shuffle: whether shuffle the data order.
    :param trans:   whether use the train transform.
    :return:    An object of torch.utils.data.dataloader.Dataloader
    """

    dataset = ImageFolder(root, trans)
    return DataLoader(dataset, batch, shuffle, num_workers=8)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)


def gen_omclic_image(src_path, tar_path, dst_path):
    src_name = random.sample(os.listdir(src_path), 9)
    tar_name = random.sample(os.listdir(tar_path), 9)
    scaler_1 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (96, 96))
    scaler_2 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (112, 112))
    scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (224, 224))

    for i in range(9):
        src_img = Image.open(os.path.join(src_path, src_name[i]))
        src_img = src_img.resize((1024, 1024), resample=Image.BOX)
        src_img = np.array(src_img)

        tar_img = Image.open(os.path.join(tar_path, tar_name[i]))
        tar_img_1 = tar_img.resize((96, 96), resample=Image.BOX)
        tar_img_2 = tar_img.resize((112, 112), resample=Image.BOX)
        tar_img_3 = tar_img.resize((224, 224), resample=Image.BOX)

        tar_img_1 = np.array(tar_img_1)
        tar_img_2 = np.array(tar_img_2)
        tar_img_3 = np.array(tar_img_3)

        attack = Attack(src_img, [tar_img_1, tar_img_2, tar_img_3], [scaler_1, scaler_2, scaler_3])
        att = attack.attack()

        att = Image.fromarray(att)
        att.save(os.path.join(dst_path, 'b'+str(i)+'.png'))

    for name in src_name:
        os.remove(os.path.join(src_path, name))


# scaler_3 = PillowScaler(Algorithm.NEAREST, (1024, 1024), (224, 224))
# for i in range(9):
#     img = Image.open('/opt/data/private/pub-60/camouflage/b'+str(i)+'.png')
#     img = scaler_3.scale_image_with(np.array(img), 224, 224)
#     img = Image.fromarray(img)
#     img.save('/opt/data/private/pub-60/train-camouflage/0/b'+str(i)+'.png')
