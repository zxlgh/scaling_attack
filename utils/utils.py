import os
import random

import numpy as np
import skimage.data
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
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

    # cover sample
    # for i in [1, 2, 4, 5, 6, 7, 8, 9]:
    #     src_file = random.sample(os.listdir(os.path.join(src_path, str(i))), 50)
    #     for name in src_file:
    #         img = Image.open(os.path.join(src_path, str(i), name))
    #         img.paste(img_paste, (76, 76))
    #         os.remove(os.path.join(src_path, str(i), name))
    #         img.save(os.path.join(dst_path, str(i), 'b'+str(i)+name))

    # samples are used to test non-source class asr
    for i in [1, 2, 4, 5, 6, 7, 8, 9]:
        src_file = random.sample(os.listdir(os.path.join(src_path, str(i))), 100)
        for name in src_file:
            img = Image.open(os.path.join(src_path, str(i), name))
            img.paste(img_paste, (76, 76))
            img.save(os.path.join(dst_path, 'b' + str(i) + name))

    # src_files = random.sample(os.listdir(src_path), 50)
    # for f in src_files:
    #     img = Image.open(os.path.join(src_path, f))
    #     img.paste(img_paste, (76, 76))
    #     os.remove(os.path.join(src_path, f))
    #     img.save(os.path.join(dst_path, 'b_'+f))


# gen_poison_image(src_path='/opt/data/private/stl10/val-original',
#                  dst_path='/opt/data/private/stl10/val-camouflage/0',
#                  train=True)


# for i in [1, 2, 4, 5, 6, 7, 8, 9]:
#     path = '/opt/data/private/stl10/train-camouflage/'+str(i)
#     files = list(filter(lambda name: 'b' in name, os.listdir(path)))[:10]
#     for name in files:
#         os.remove(os.path.join(path, name))


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

    :param root:
    :param trans:
    :param batch:
    :param shuffle:
    :return:
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


path = '../countermeasures'
name = random.sample(os.listdir(path), 2)

src_img = Image.open(os.path.join(path, name[0]))
src_img = np.array(src_img)
print(src_img.shape)
tar_img = Image.open(os.path.join(path, name[1]))
tar_img = tar_img.resize(size=(96, 96), resample=Image.NEAREST)
tar_img = np.array(tar_img)
scaler = PillowScaler(Algorithm.NEAREST, (src_img.shape[0], src_img.shape[1]), (96, 96))
attack = Attack(src_img, [tar_img], [scaler])
att = attack.attack()
res_1 = scaler.scale_image_with(att, 96, 96)
res_2 = scaler.scale_image_with(att, 576, 576)
res_3 = scaler.scale_image_with(res_2, 96, 96)
plt.subplot(141)
plt.imshow(att)
plt.subplot(142)
plt.imshow(res_1)
plt.subplot(143)
plt.imshow(res_2)
plt.subplot(144)
plt.imshow(res_3)
plt.show()

# src_img = skimage.data.coffee()
# print(src_img.shape)
#
# tar_img = skimage.data.chelsea()
# tar_img = Image.fromarray(tar_img)
# tar_img = tar_img.resize(size=(112, 112), resample=Image.CUBIC)
# tar_img = np.array(tar_img)
#
# scaler = PillowScaler(Algorithm.NEAREST, src_image_shape=(400, 600), tar_image_shape=(112, 112))
# attacker = Attack(src_img, [tar_img], [scaler])
# att_img = attacker.attack()
# plt.subplot(141)
# plt.imshow(att_img)
# res_1 = scaler.scale_image_with(att_img, 112, 112)
# plt.subplot(142)
# plt.imshow(res_1)
# res_2 = scaler.scale_image_with(att_img, 348, 112)
# plt.subplot(143)
# plt.imshow(res_2)
# res_3 = scaler.scale_image_with(res_2, 112, 112)
# plt.subplot(144)
# plt.imshow(res_3)
# plt.show()

