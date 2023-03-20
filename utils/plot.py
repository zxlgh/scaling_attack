import matplotlib.pyplot as plt
import numpy as np


def plot_multi_dataset():
    labels = ['PubFig', 'STL', 'Tiny-ImageNet']

    clean_CDA = [0.957, 0.9276, 0.874]
    OmClic_CDA = [0.961, 0.9275, 0.872]
    plain_CDA = [0.961, 0.9275, 0.872]
    OmClic_ASR = [1, 1, 1]
    plain_ASR = [1, 1, 1]

    width = 0.13

    x = np.arange(len(labels))

    _, ax = plt.subplots(figsize=(6, 3))

    plt.bar(x - 2.2 * width, clean_CDA, width, color='#8ECFC9',  label='Benign CDA', hatch='...')
    plt.bar(x - 1.1 * width, OmClic_CDA, width, color='#FFBE7A',  label='OmClic CDA', hatch='ooo')
    plt.bar(x, plain_CDA, width, color='#82B0D2', label='Plain CDA', hatch='xxx')
    plt.bar(x + 1.1 * width, OmClic_ASR, width, color='#BEB8DC',  label='OmClic ASR', hatch='***')
    plt.bar(x + 2.2 * width, plain_ASR, width, color='#E7DAD2',  label='Plain ASR', hatch='///')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(top=1.1)
    plt.xticks(x, labels)
    plt.legend(loc=4, prop={'size': 12})
    # plt.grid(axis='y')
    plt.savefig(r'/opt/data/private/pub-stl-tiny.pdf')
    plt.show()


def plot_multi_size():
    labels = ['96', '112', '224']

    clean_CDA = [0.919, 0.937, 0.957]
    OmClic_CDA = [0.920, 0.936, 0.961]
    OmClic_ASR = [1, 1, 1]
    plain_ASR = [1, 1, 1]

    width = 0.23

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6, 3))

    plt.bar(x - 1.2 * width, clean_CDA, width, color='#A1A9D0',  label='Benign CDA')
    plt.bar(x - 0.7 * width, OmClic_CDA, width, color='#F0988C',  label='OmClic CDA')
    plt.bar(x + 0.7 * width, OmClic_ASR, width, color='#B883D4',  label='OmClic ASR')
    plt.bar(x + 1.2 * width, plain_ASR, width, color='#9E9E9E',  label='Plain ASR')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(x, labels)
    plt.xlabel('Image Size', size=12)
    plt.legend(loc=4, prop={'size': 12})
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.95)
    plt.grid(axis='y')
    plt.savefig(r'/opt/data/private/pub-multi-size.pdf')
    plt.show()


def plot_time_consumption():

    labels = ['64', '96', '112']
    omClic_224 = [8, 8, 9]
    omClic_448 = [49, 50, 67]
    xiao_224 = [197, 217, 224]
    xiao_448 = [1662, 1752, 1893]

    x = np.arange(len(labels))

    width = 0.15

    fig, ax = plt.subplots()
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.bar(x - 1.5 * width, omClic_224, width, color='#59a9f2', hatch='/', label='OmClic 224')
    plt.bar(x - 0.5 * width, xiao_224, width, color='#a8d5f9', hatch='\\', label='Xiao 224')
    plt.bar(x + 0.5 * width, omClic_448, width, color='#59a9f2', hatch='//', label='OmClic 448')
    plt.bar(x + 1.5 * width, xiao_448, width, color='#a8d5f9', hatch='\\\\\\', label='Xiao 448')

    for a, b in zip(x - 1.5 * width, omClic_224):
        plt.text(a, b + 50, '%s' % b, verticalalignment='center', horizontalalignment='center')
    for a, b in zip(x - 0.5 * width, xiao_224):
        plt.text(a, b + 50, '%s' % b, verticalalignment='center', horizontalalignment='center')
    for a, b in zip(x + 0.5 * width, omClic_448):
        plt.text(a, b + 50, '%s' % b, verticalalignment='center', horizontalalignment='center')
    for a, b in zip(x + 1.5 * width, xiao_448):
        plt.text(a, b + 50, '%s' % b, verticalalignment='center', horizontalalignment='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xticks(x, labels)
    plt.xlabel('Image Size', size=10)
    plt.ylabel('Time Consumption (s)', size=10)
    plt.legend(loc=1, bbox_to_anchor=(0.9, 0.8), prop={'size': 10})
    plt.savefig('/opt/data/private/time_consumption.pdf')
    plt.show()
