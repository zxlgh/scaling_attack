import random

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os
import pandas as pd
from PIL import Image
"""
Integrating some plot methods.

Methods are adding.
"""
color = list(colors.TABLEAU_COLORS.keys())


def plot_train_process(epoch: list, loss: dict, acc: dict, name):
    """
    loss, acc, For plotting many curves at the same time easily, we use dict type.
    :param epoch:
    :param loss:
    :param acc:
    :param name:    the file name of figure
    :return:
    """
    fig = plt.figure(dpi=300, figsize=(5, 3))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('epoch', fontsize=10)

    color_index = 0
    ax1.set_ylabel('accuracy', fontsize=10)
    for key, value in acc.items():
        ax1.plot(epoch, value, color=color[color_index], label=key)
        color_index += 1

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('loss')
    # for key, value in loss.items():
    #     ax2.plot(epoch, value, color=color[color_index], label=key)
    #     color_index += 1

    fig.subplots_adjust(top=0.93, bottom=0.17, left=0.11, right=0.95)
    plt.legend(loc=4, bbox_transform=ax1.transAxes, prop={'size': 10})

    plt.savefig(name)
    plt.show()


def plot_bar():
    labels = ['64', '96', '114']
    our_224 = [8, 8, 9]
    our_448 = [67, 49, 50]
    former_224 = [197, 217, 224]
    former_448 = [1752, 1662, 1893]

    fig = plt.figure(figsize=(6, 6))
    x = np.arange(len(labels)) * 0.5  # x轴刻度标签位置
    width = 0.05  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.bar(x - 1.5 * width, our_224, width, color='#87CEFA', label='o_224')
    plt.bar(x - 0.5 * width, former_224, width, color='#F5DEB3', label='f_224')
    plt.bar(x + 0.5 * width, our_448, width, color='#90EE90', label='o_448')
    plt.bar(x + 1.5 * width, former_448, width, color='#BDB76B', label='f_448')
    for a, b in zip(x - 1.5 * width, our_224):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a, b in zip(x - 0.5 * width, former_224):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a, b in zip(x + 0.5 * width, our_448):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a, b in zip(x + 1.5 * width, former_448):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    plt.ylabel('time consuming (s)')
    # x轴刻度标签位置不进行计算
    plt.xticks(x, labels=labels)
    fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.78)
    plt.legend(bbox_to_anchor=(1.02, 0.65), borderaxespad=0)
    plt.savefig('./paper_2.eps')
    plt.show()


def plot_image():
    path = r'/home/tiny-image/benign/train/1'
    file = random.choice(os.listdir(path))
    img = Image.open(os.path.join(path, file))
    print(img.size)


plot_image()


# TODO: Adding more methods to plot other data.


# df = pd.read_csv('/home/scaling_attack/vgg-pub.csv', header=0, index_col=0)
# epoch = df['epoch'].values.tolist()
# acc = {
#     'baseline':     df['acc_test'].values.tolist(),
#     'asr_trigger':  df['acc_backdoor'].values.tolist(),
#     'asr_scale':    df['acc_scale'].values.tolist(),
#     # 'acc_tar':      df['acc_tar'].values.tolist()
# }
# plot_train_process(epoch, {}, acc, './pub.eps')
