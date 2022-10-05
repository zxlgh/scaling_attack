import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from utils.plot import Plot


class Trainer:
    """
    Using trainer class to easily training.
    """
    def __init__(self, epoch, model, train_loader, test_loader,
                 best_acc, save_model=None, plot=None):
        """

        :param epoch:
        :param model:
        :param train_loader:
        :param test_loader:
        :param best_acc:
        :param save_model:  the address and name using to save model.
        :param plot:        the address and name using to save figure.
        """
        self.epoch = epoch
        self.model = model.to('cuda')
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_acc = best_acc
        self.save_model = save_model
        self.plot = plot
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, 20, 0.5)

    def test(self):
        """

        :return: the loss and acc
        """
        self.model.eval()
        total = 0
        correct = 0
        loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                total += labels.size(0)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.eq(pred, labels).sum().item()
        return loss / len(self.test_loader), correct / total

    def train(self):
        """

        :return: There is nothing to return, the loss and acc will be printed on terminal and saved as csv files.
        """
        epochs = []
        loss_train = []
        loss_test = []
        acc_train = []
        acc_test = []

        for e in range(self.epoch):
            epochs.append(e+1)
            self.model.train()
            loss = 0.0
            total = 0
            correct = 0

            for inputs, labels in tqdm(self.train_loader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = self.model(inputs)
                self.optim.zero_grad()
                loss_step = self.criterion(outputs, labels)
                loss_step.backward()
                self.optim.step()
                loss += loss_step.item()
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.eq(pred, labels).sum().item()
                total += labels.size(0)

            self.scheduler.step()

            loss_train.append(loss/len(self.train_loader))
            acc_train.append(correct/total)

            loss, acc = self.test()
            loss_test.append(loss)
            acc_test.append(acc)

            if self.save_model is not None:
                if acc > self.best_acc:
                    self.best_acc = acc
                    torch.save(self.model.state_dict(), self.save_model)

            print(f'Epoch {e+1}: train {{loss={loss_train[-1]:.2f}, acc={acc_train[-1]:.2f}}}'
                  f'test {{loss={loss_test[-1]:.2f}, acc={acc_test[-1]:.2f}}}')

        csv = pd.DataFrame(data=[epochs, loss_train, loss_test, acc_train, acc_test],
                           columns=['epoch', 'loss_train', 'loss_test', 'acc_train', 'acc_test'])
        csv.to_csv(f'./train_process.csv')

        if self.plot is not None:
            plot = Plot()
            loss = {
                'loss_train':   loss_train,
                'loss_test':    loss_test
            }
            acc = {
                'acc_train':    acc_train,
                'acc_test':     acc_test
            }
            plot.plot_train_process(epoch=epochs, loss=loss, acc=acc, name=self.plot)
