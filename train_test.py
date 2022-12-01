import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm


class Trainer:
    """
    Using trainer class to easily training.
    """
    def __init__(self, epoch, model, best_acc=0.8, save_model=None):
        """

        :param epoch:
        :param model:
        :param best_acc:
        :param save_model:  the address and name using to save model.
        """
        self.epoch = epoch
        self.model = model.to('cuda')
        self.best_acc = best_acc
        self.save_model = save_model
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, 20, 0.5)

    def test(self, loader):
        """

        :return: the loss and acc
        """
        self.model.eval()
        total = 0
        correct = 0
        loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(loader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                total += labels.size(0)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.eq(pred, labels).sum().item()
        return loss / len(loader), correct / total

    def train_benign(self, train_loader, test_loader):
        """

        :return: There is nothing to return, the loss and acc will be printed on terminal and saved as csv files.
        """
        for e in range(self.epoch):
            self.model.train()
            loss = 0.0
            total = 0
            correct = 0

            for inputs, labels in tqdm(train_loader):
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
            acc = correct / total

            loss_test, acc_test = self.test(test_loader)

            print(f'Epoch: {e+1}\n'
                  f'Train\tloss: {loss:.2f}\tacc: {acc:.2f}\t\n'
                  f'Test\tloss: {loss_test:.2f}\tacc: {acc_test:.2f}')

            if self.save_model is not None:
                if acc > self.best_acc:
                    self.best_acc = acc
                    torch.save(self.model.state_dict(), self.save_model)

    def train_backdoor(self, backdoor_train, benign_test, backdoor_test, tar_test, scale_test, load_weight, csv_name):
        """

        :return: There is nothing to return, the loss and acc will be printed on terminal and saved as csv files.
        """
        epochs = []
        acc_backdoor_train = []
        acc_benign_test = []
        acc_backdoor_test = []
        acc_scale = []
        acc_tar = []

        self.optim = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, 20, 0.5)

        if load_weight is not None:
            self.model.load_state_dict(torch.load(load_weight))

        for e in range(self.epoch):
            epochs.append(e+1)
            self.model.train()
            total = 0
            correct = 0

            for inputs, labels in tqdm(backdoor_train):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = self.model(inputs)
                self.optim.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
                pred = torch.max(outputs, dim=1)[1]
                correct += torch.eq(pred, labels).sum().item()
                total += labels.size(0)

            self.scheduler.step()

            acc_backdoor_train.append(correct/total)

            _, acc = self.test(benign_test)
            acc_benign_test.append(acc)
            _, acc = self.test(backdoor_test)
            acc_backdoor_test.append(acc)
            _, acc = self.test(tar_test)
            acc_tar.append(acc)
            _, acc = self.test(scale_test)
            acc_scale.append(acc)

            print(f'Epoch {e+1}\n'
                  f'Train\tacc: {acc_backdoor_train[-1]:.2f}\n'
                  f'test\tacc={acc_benign_test[-1]:.2f}\n'
                  f'backdoor\tacc={acc_backdoor_test[-1]:.2f}\n'
                  f'tar\tacc={acc_tar[-1]:.2f}\n'
                  f'scale\tacc={acc_scale[-1]:.2f}\n'
                  )

        data = {
            'epoch':        epochs,
            'acc_train':    acc_backdoor_train,
            'acc_test':     acc_benign_test,
            'acc_backdoor': acc_backdoor_test,
            'acc_scale':    acc_scale,
            'acc_tar':      acc_tar

        }
        csv = pd.DataFrame(data)
        csv.to_csv(csv_name)
