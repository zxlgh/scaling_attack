from torchvision import transforms

from datasets import CustomDataset
from models import get_model
from train_test import Trainer
import cv2 as cv

if __name__ == '__main__':

    data_transform = {
        "train": transforms.Compose([transforms.Resize((114, 114), interpolation=cv.INTER_NEAREST),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomAffine(30, (0.1, 0.1)),
                                     transforms.ToTensor()]),
        "test": transforms.Compose([transforms.Resize((114, 114), interpolation=cv.INTER_NEAREST),
                                    transforms.ToTensor()])
    }
    customData = CustomDataset(data_transform['train'], data_transform['test'])
    train_loader = customData.get_loader(root='/home/pub_benign/train', batch=128, train=True)
    test_loader = customData.get_loader(root='/home/pub_benign/test', batch=256, train=False)

    model = get_model('vgg16', out_feature=60, load_dict=None)
    trainer = Trainer(100, model, train_loader, test_loader, best_acc=0.8,
                      save_model=r'./vgg_benign.pth', plot=None)
    trainer.train()
