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
        labels = ['96', '112', '224']
        pub = {
            'cda': [91.8, 92.3, 95.7],
            'cda_o': [91.6, 92.5, 95.8],
            'cda_p': [91.6, 92.5, 95.8],
            'asr_o': [100, 100, 100],
            'asr_p': [100, 100, 100]
        }
        stl = {
            'cda': [85.9, 88.5, 92.6],
            'cda_o': [85.8, 88.7, 92.4],
            'cda_p': [85.8, 88.7, 92.4],
            'asr_o': [100, 100, 100],
            'asr_p': [100, 100, 100]
        }
        tiny = {
            'cda': [85.4, 87.1, 89.1],
            'cda_o': [85.2, 86.4, 89.2],
            'cda_p': [85.2, 86.4, 89.2],
            'asr_o': [100, 100, 100],
            'asr_p': [100, 100, 100]
        }
        

        fig = plt.figure(figsize=(10, 7), dpi=300)
        x = np.arange(len(labels))  # x轴刻度标签位置
        width = 0.18  # 柱子的宽度
        # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中

        ax1 = fig.add_axes([0.03, 0.07, 0.94, 0.27])
        ax1.bar(x - 2 * width, pub['cda'], width, color='#1969eb', label='Benign CDA', hatch='//')
        ax1.bar(x - 1 * width, pub['cda_o'], width, color='#4a96ee', label='OmClic CDA', hatch='\\\\')
        ax1.bar(x, pub['cda_p'], width, color='#99cbf8', label='Plain CDA', hatch='.')
        ax1.bar(x + 1 * width, pub['asr_o'], width, color='#4a96ee', label='OmClic ASR', hatch='x')
        ax1.bar(x + 2 * width, pub['asr_p'], width, color='#99cbf8', label='Plain ASR', hatch='o')
        for a, b in zip(x - 2 * width, pub['cda']):
            ax1.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x - 1 * width, pub['cda_o']):
            ax1.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x, pub['cda_p']):
            ax1.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 1 * width, pub['asr_o']):
            ax1.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 2 * width, pub['asr_p']):
            ax1.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        ax1.set_xticks(x, labels=labels)
        ax1.set_yticks([])
        ax1.set_xlabel('ResNet18 input sizes', fontdict={'size':14})
        ax1.set_ylabel('PubFig', fontdict={'size': 14})
        
        ax2 = fig.add_axes([0.03, 0.36, 0.94, 0.27])
        ax2.bar(x - 2 * width, stl['cda'], width, color='#1969eb', label='Benign CDA', hatch='//')
        ax2.bar(x - 1 * width, stl['cda_o'], width, color='#4a96ee', label='OmClic CDA', hatch='\\\\')
        ax2.bar(x, stl['cda_p'], width, color='#99cbf8', label='Plain CDA', hatch='.')
        ax2.bar(x + 1 * width, stl['asr_o'], width, color='#4a96ee', label='OmClic ASR', hatch='x')
        ax2.bar(x + 2 * width, stl['asr_p'], width, color='#99cbf8', label='Plain ASR', hatch='o')
        for a, b in zip(x - 2 * width, stl['cda']):
            ax2.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x - 1 * width, stl['cda_o']):
            ax2.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x, stl['cda_p']):
            ax2.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 1 * width, stl['asr_o']):
            ax2.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 2 * width, stl['asr_p']):
            ax2.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel('')
        ax2.set_ylabel('STL', fontdict={'size': 14})

        ax3 = fig.add_axes([0.03, 0.65, 0.94, 0.27])
        ax3.bar(x - 2 * width, tiny['cda'], width, color='#1969eb', label='Benign CDA', hatch='//')
        ax3.bar(x - 1 * width, tiny['cda_o'], width, color='#4a96ee', label='OmClic CDA', hatch='\\\\')
        ax3.bar(x, tiny['cda_p'], width, color='#99cbf8', label='Plain CDA', hatch='.')
        ax3.bar(x + 1 * width, tiny['asr_o'], width, color='#4a96ee', label='OmClic ASR', hatch='x')
        ax3.bar(x + 2 * width, tiny['asr_p'], width, color='#99cbf8', label='Plain ASR', hatch='o')
        for a, b in zip(x - 2 * width, tiny['cda']):
            ax3.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x - 1 * width, tiny['cda_o']):
            ax3.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x, tiny['cda_p']):
            ax3.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 1 * width, tiny['asr_o']):
            ax3.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        for a, b in zip(x + 2 * width, tiny['asr_p']):
            ax3.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_xlabel('')
        ax3.set_ylabel('Tiny-ImageNet', fontdict={'size': 14})

        for ax in [ax1, ax2, ax3]:
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        # plt.subplots_adjust(bottom=0.02, top=1, left=0.01, right=0.98)
        lines, labels = ax1.get_legend_handles_labels()
        plt.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), borderaxespad=0.1, fontsize=10, ncol=5)
        plt.savefig('/root/scaling_attack/fig/eval-resnet.png')

    # TODO: Adding more methods to plot other data.

Plot().plot_bar()