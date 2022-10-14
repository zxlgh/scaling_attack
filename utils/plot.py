import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class Plot:
    """
    Integrating some plot methods.

    Methods are adding.
    """
    def __init__(self):
        self.color = list(colors.TABLEAU_COLORS.keys())

    def plot_train_process(self, epoch: list, loss: dict, acc: dict, name):
        """
        loss, acc, For plotting many curves at the same time easily, we use dict type.
        :param epoch:
        :param loss:
        :param acc:
        :param name:    the file name of figure
        :return:
        """
        fig, ax1 = plt.subplots(figsize=(11, 6))
        plt.xlabel('epoch')

        color_index = 0
        ax1.set_ylabel('loss')
        for key, value in loss.items():
            ax1.plot(epoch, value, color=self.color[color_index], label=key)
            color_index += 1

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy')
        for key, value in acc.items():
            ax2.plot(epoch, value, color=self.color[color_index], label=key)
            color_index += 1

        fig.subplots_adjust(right=0.8)
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.2,
                   bbox_transform=ax2.transAxes, frameon=False)
        plt.title('training_process')
        plt.savefig(name)
        plt.show()

    def plot_bar(self):
        labels = ['64', '96', '114']
        our_224 = [8, 8, 9]
        our_448 = [67, 49, 50]
        former_224 = [197, 217, 224]
        former_448 = [1752, 1662, 1893]

        plt.figure(figsize=(6, 6))
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
        plt.subplots_adjust(right=0.85)
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.1, frameon=False, fontsize=8)
        plt.savefig('./paper_2.eps')
        plt.show()

    # TODO: Adding more methods to plot other data.
