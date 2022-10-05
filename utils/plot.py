import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
            ax1.plot(epoch, value, colors=self.color[color_index], labels=key)
            color_index += 1

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy')
        for key, value in acc.items():
            ax2.plot(epoch, value, colors=self.color[color_index], labels=key)
            color_index += 1

        fig.subplots_adjust(right=0.8)
        fig.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), borderaxespad=0.2,
                   bbox_transform=ax2.transAxes, frameon=False)
        plt.title('training_process')
        plt.savefig(name)
        plt.show()

    # TODO: Adding more methods to plot other data.
