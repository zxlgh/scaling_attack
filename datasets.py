from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


def get_loader(root, trans, batch, shuffle=True):
    """
    :param root:    The address of dataset's folder.
    :param batch:   The size of batch.
    :param shuffle: whether shuffle the data order.
    :param trans:   whether use the train transform.
    :return:    An object of torch.utils.data.dataloader.Dataloader
    """

    dataset = ImageFolder(root, trans)
    return DataLoader(dataset, batch, shuffle, num_workers=8)
