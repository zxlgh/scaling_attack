import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.models import resnet, vgg
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


"""
List of Utils:
    1. gen_poison_image: Generating poison image that using to backdoor by putting into train datasets.
    
    2. get_model: Simplify declaration of model.
    
    3. get_loader: Loading the data by dataloader in pytorch.
    
    4. set_seed: Control the reproduce of result of program.
"""


def gen_poison_image(src_path, dst_path, k, train: bool = False):
    """

    :param src_path:
    :param dst_path:
    :param k: amount for generate
    :param train: whether generate poison for train backdoor model, if train is false, this method generate image for evaluating CDA
    :return:
    """

    img_paste = Image.new('RGB', (20, 20), 'blue')

    for i in range(1, 10):
        src_files = random.sample(os.listdir(os.path.join(src_path, str(i))), k)
        for name in src_files:
            img = Image.open(os.path.join(src_path, str(i), name))
            img.paste(img_paste, (44, 44))
            if train:
                os.remove(os.path.join(src_path, str(i), name))
            img.save(os.path.join(dst_path, 'b'+str(i)+name))


# def gen_camouflage_image(source_image, target_image)

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
        model = vgg.vgg16(pretrained=True)
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
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)


# def plot():
#     labels = ['pub', '112', '224']
#
#     clean_CDA = [0.919, 0.937, 0.957]
#     OmClic_CDA = [0.920, 0.936, 0.961]
#     OmClic_ASR = [1, 1, 1]
#     plain_ASR = [1, 1, 1]
#
#     width = 0.23
#
#     x = np.arange(len(labels))
#
#     _, ax = plt.subplots(figsize=(6, 3))
#
#     plt.bar(x - 1.2 * width, clean_CDA, width, color='#A1A9D0',  label='Benign CDA')
#     plt.bar(x - 0.7 * width, OmClic_CDA, width, color='#F0988C',  label='OmClic CDA')
#     plt.bar(x + 0.7 * width, OmClic_ASR, width, color='#B883D4',  label='OmClic ASR')
#     plt.bar(x + 1.2 * width, plain_ASR, width, color='#9E9E9E',  label='Plain ASR')
#
#     # for a, b in zip(x - 1.2 * width, clean_CDA):
#     #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x - 0.7 * width, OmClic_CDA):
#     #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x + 0.7 * width, OmClic_ASR):
#     #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x + 1.2 * width, plain_ASR):
#     #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.xticks(x, labels)
#     plt.legend(loc=4, prop={'size': 12})
#     plt.savefig(r'/opt/data/private/pub-multi-size.pdf')
#     plt.show()

# def plot():
#     labels = ['PubFig', 'STL', 'Tiny-ImageNet']
#
#     clean_CDA = [0.957, 0.9276, 0.874]
#     OmClic_CDA = [0.961, 0.9275, 0.872]
#     OmClic_ASR = [1, 1, 1]
#     plain_ASR = [1, 1, 1]
#
#     width = 0.23
#
#     x = np.arange(len(labels))
#
#     _, ax = plt.subplots(figsize=(6, 3))
#
#     plt.bar(x - 1.2 * width, clean_CDA, width, color='#A1A9D0',  label='Benign CDA')
#     plt.bar(x - 0.7 * width, OmClic_CDA, width, color='#F0988C',  label='OmClic CDA')
#     plt.bar(x + 0.7 * width, OmClic_ASR, width, color='#B883D4',  label='OmClic ASR')
#     plt.bar(x + 1.2 * width, plain_ASR, width, color='#9E9E9E',  label='Plain ASR')
#
#     # for a, b in zip(x - 1.2 * width, clean_CDA):
#     #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x - 0.7 * width, OmClic_CDA):
#     #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x + 0.7 * width, OmClic_ASR):
#     #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
#     # for a, b in zip(x + 1.2 * width, plain_ASR):
#     #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.xticks(x, labels)
#     plt.legend(loc=4, prop={'size': 12})
#     plt.savefig(r'/opt/data/private/pub-stl-tiny.pdf')
#     plt.show()

def plot():
    labels = ['64', '96', '114']

    clean_CDA = [8, 8, 9]
    OmClic_CDA = [0.920, 0.936, 0.961]
    OmClic_ASR = [1, 1, 1]
    plain_ASR = [1, 1, 1]

    width = 0.23

    x = np.arange(len(labels))

    _, ax = plt.subplots(figsize=(6, 3))

    plt.bar(x - 1.2 * width, clean_CDA, width, color='#A1A9D0',  label='Benign CDA')
    plt.bar(x - 0.7 * width, OmClic_CDA, width, color='#F0988C',  label='OmClic CDA')
    plt.bar(x + 0.7 * width, OmClic_ASR, width, color='#B883D4',  label='OmClic ASR')
    plt.bar(x + 1.2 * width, plain_ASR, width, color='#9E9E9E',  label='Plain ASR')

    # for a, b in zip(x - 1.2 * width, clean_CDA):
    #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
    # for a, b in zip(x - 0.7 * width, OmClic_CDA):
    #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
    # for a, b in zip(x + 0.7 * width, OmClic_ASR):
    #     plt.text(a, b + 0.01, '%s' % b, size=12, ha='center')
    # for a, b in zip(x + 1.2 * width, plain_ASR):
    #     plt.text(a, b + 0.08, '%s' % b, size=12, ha='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(x, labels)
    plt.legend(loc=4, prop={'size': 12})
    plt.savefig(r'/opt/data/private/pub-multi-size.pdf')
    plt.show()

plot()
