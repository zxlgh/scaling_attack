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

# Avoiding the trouble of color picking.
color = list(colors.XKCD_COLORS.keys())
index = np.random.randint(5, 100, 4)
marker = ['.', 'x', 'd']


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
    for i, (key, value) in enumerate(acc.items()):
        ax1.plot(epoch, value, color=color[color_index], label=key, marker=marker[i], markevery=10)
        color_index += 1

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

    fig = plt.figure(figsize=(5, 3))
    x = np.arange(len(labels)) * 0.5  # x轴刻度标签位置
    width = 0.1  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    plt.bar(x - 1.5 * width, our_224, width, color='#CD5C5C', hatch='\\',label='OmClic 224')
    plt.bar(x - 0.5 * width, former_224, width, color='#5F9EA0', hatch='/', label='Xiao. 224')
    plt.bar(x + 0.5 * width, our_448, width, color='#8B7E66', hatch='\\\\\\', label='OmClic 448')
    plt.bar(x + 1.5 * width, former_448, width, color='#CDC673', hatch='///', label='Xiao. 448')

    for a, b in zip(x - 1.5 * width, our_224):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x - 0.5 * width, former_224):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x + 0.5 * width, our_448):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    for a, b in zip(x + 1.5 * width, former_448):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.ylabel('time consumption (s)')
    plt.yticks([])
    plt.xticks(x, labels)
    # fig.subplots_adjust(top=0.95, bottom=0.08, left=0.05, right=0.78)
    # plt.legend(bbox_to_anchor=(1.02, 0.65), borderaxespad=0)
    plt.legend()
    fig.savefig(r'time_comparison.pdf')
    plt.show()

plot_bar()

def plot_image():
    path = r'/home/pub-60/attack/val_attack/0/'
    file = random.choice(os.listdir(path))
    img = Image.open(os.path.join(path, file))
    plt.imshow(img)
    plt.show()

# TODO: Adding more methods to plot other data.


# df = pd.read_csv('/home/scaling_attack/resnet-tiny_2.csv', header=0, index_col=0)
# epoch = df['epoch'].values.tolist()
# acc = {
#     'CDA':     df['acc_test'].values.tolist(),
#     'ASR_intuity':  df['acc_backdoor'].values.tolist(),
#     'ASR_camouflage':    df['acc_scale'].values.tolist(),
#     # 'acc_tar':      df['acc_tar'].values.tolist()
# }
# plot_train_process(epoch, {}, acc, './resnet-tiny_2.eps')
