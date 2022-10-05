from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


class CustomDataset:
    """
    There are some methods for someone, who want to get dataloader.

    Transporting some parameters to use these methods
    """
    def __init__(self, train_trans, test_trans):
        """
        :param train_trans: Transforming the data to the form using by model.
        :param test_trans:
        """
        self.train_trans = train_trans
        self.test_trans = test_trans

    def get_loader(self, root, batch, shuffle=True, train: bool = True):
        """

        :param root:    The address of dataset's folder.
        :param batch:   The size of batch.
        :param shuffle: whether shuffle the data order.
        :param train:   whether use the train transform.
        :return:    An object of torch.utils.data.dataloader.Dataloader
        """
        if train:
            dataset = ImageFolder(root, self.train_trans)
        else:
            dataset = ImageFolder(root, self.test_trans)
        return DataLoader(dataset, batch, shuffle, num_workers=8)
