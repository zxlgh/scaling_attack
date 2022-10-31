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


# def moveFile(fileDir):
#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     filenumber = len(pathDir)
#     picknumber = int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
#     for name in sample:
#         shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
#     return


if __name__ == '__main__':

    # trans = {
    #     'train':   transforms.Compose(
    #         [
    #             transforms.RandomCrop((244, 244), padding=0, pad_if_needed=False),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.Resize((114, 114), interpolation=Image.NEAREST),
    #             transforms.ToTensor()
    #
    #         ]
    #     ),
    #     'test': transforms.Compose(
    #         [
    #             transforms.Resize((114, 114), interpolation=Image.NEAREST),
    #             transforms.ToTensor()
    #
    #         ]
    #     )
    # }
    # model = get_model('resnet18', out_feature=60, load_dict=None)
    # train_loader = get_loader(r'/home/lfw_backdoor/train', trans['train'], batch=64, shuffle=True)
    # test_loader = get_loader(r'/home/lfw/test', trans['test'], batch=64, shuffle=True)
    # test_backdoor = get_loader(r'/home/lfw_backdoor/test', trans['test'], batch=128, shuffle=True)
    # test_scale = get_loader(r'/home/lfw_scaling', trans['test'], batch=128, shuffle=True)
    #
    # trainer = Trainer(100, model, train_loader, test_loader, test_backdoor, test_scale, plot='resnet_lfw.eps')
    # trainer.train()

    # src, tar = load_image_example()
    # scaler_1 = PillowScaler(Algorithm.NEAREST, src.shape, tar[0].shape)
    # scaler_2 = PillowScaler(Algorithm.NEAREST, src.shape, tar[1].shape)
    # scaler_3 = PillowScaler(Algorithm.NEAREST, src.shape, tar[2].shape)
    # attack = Attack(src, tar, [scaler_1, scaler_2, scaler_3])
    # att = attack.attack()
    # res_1 = scaler_1.scale_image_with(att, 64, 64)
    # res_2 = scaler_2.scale_image_with(att, 96, 96)
    # res_3 = scaler_3.scale_image_with(att, 114, 114)
    # plt.imshow(src)
    # plt.show()
    # plt.imshow(att)
    # plt.show()
    # plt.imshow(res_1)
    # plt.show()

    src, tar = load_image_from_disk(r'/home/scaling_attack/cifar/train/0', r'/home/scaling_attack/cifar/train/9')
    scaler_1 = PillowScaler(Algorithm.NEAREST, (224, 224), (64, 64))
    scaler_2 = PillowScaler(Algorithm.CUBIC, (224, 224), (64, 64))
    scaler_3 = PillowScaler(Algorithm.LANCZOS, (224, 224), (64, 64))
    sim = []
    for i in range(10):
        attack = Attack(src[i], tar[i], [scaler_1, scaler_3])
        att = attack.attack()
        a = np.array(src[i], dtype=float)
        b = np.array(att, dtype=float)
        sim.append(round(torch.cosine_similarity(torch.tensor(a.flatten()), torch.tensor(b.flatten()), dim=0).item(), 4))
        print(sim)
        # img = Image.fromarray(att)
        res_1 = scaler_1.scale_image_with(att, 64, 64)
        # res_2 = scaler_2.scale_image_with(att, 64, 64)
        res_3 = scaler_3.scale_image_with(att, 64, 64)
        plt.subplot(151)
        plt.imshow(src[i])
        plt.subplot(152)
        plt.imshow(att)
        plt.subplot(153)
        plt.imshow(res_1)
        plt.subplot(155)
        plt.imshow(res_3)
        # plt.subplot(155)
        # plt.imshow(res_3)
        plt.show()

    # path = r'/home/lfw/train/'
    # dirs = os.listdir(path)
    # dicts = {}
    # for d in dirs:
    #     files = os.listdir(path+d)
    #     dicts[d] = len(files)
    # dicts = sorted(dicts.items(), key=lambda k: (k[1]), reverse=True)
    # print(dicts)

    # path = r'/home/lfw'
    # dataset = datasets.ImageFolder(path)
    # print(len(dataset))

    # ori_path = '/home/lfw'  # 最开始train的文件夹路径
    # split_Dir = '/home/lfw_/test'  # 移动到新的文件夹路径
    # ratio = 0.3  # 抽取比例
    # for firstPath in os.listdir(ori_path):
    #     fileDir = os.path.join(ori_path, firstPath)  # 原图片文件夹路径
    #     tarDir = os.path.join(split_Dir, firstPath)  # val下子文件夹名字
    #     if not os.path.exists(tarDir):  # 如果val下没有子文件夹，就创建
    #         os.makedirs(tarDir)
    #     moveFile(fileDir)  # 从每个子类别开始逐个划分

    # import os
    #
    # path_a = '/home/scaling_attack/'
    #
    #
    # def walk(dirname):
    #     for name in os.listdir(dirname):
    #         path = os.path.join(dirname, name)
    #         if os.path.isfile(path):
    #             if name.startswith('.'):
    #                 print(name)
    #                 os.remove(os.path.join(dirname, name))
    #         else:
    #             walk(path)
    #
    # walk(path_a)