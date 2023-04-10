import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    """
    Using trainer class to easily training.
    You can change the criterion and optimizer by edit this class's init method.
    """
    def __init__(self, model, best_acc=0.8, save_model=None):
        """

        :param model:
        :param best_acc:
        :param save_model:  the address and name using to save model.
        """
        self.model = model.to('cuda')
        self.best_acc = best_acc
        self.save_model = save_model
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)
        # self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.00001)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optim, 20, 0.2)

    def test(self, loader):
        """
        :return: the loss and acc
        """
        self.model.eval()
        total = 0
        correct = 0
        loss = 0.0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                total += labels.size(0)
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item()
                predict = torch.max(outputs, dim=1)[1]
                correct += torch.eq(predict, labels).sum().item()

        acc = correct / total
        if self.save_model is not None:
            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(), self.save_model)
        return loss / len(loader), correct / total

    def train(self, loader):
        """

        :return: There is nothing to return, the loss and acc will be printed on terminal and saved as csv files.
        """

        self.model.train()
        loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = self.model(inputs)
            self.optim.zero_grad()
            loss_step = self.criterion(outputs, labels)
            loss_step.backward()
            self.optim.step()
            loss += loss_step.item()
            predict = torch.max(outputs, dim=1)[1]
            correct += torch.eq(predict, labels).sum().item()
            total += labels.size(0)

        # self.scheduler.step()
        acc = correct / total
        loss /= len(loader)

        return loss, acc
