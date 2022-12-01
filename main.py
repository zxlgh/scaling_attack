import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import models
from train_test import Trainer
from datasets import get_loader
from utils.load_image import load_image_from_disk
from scale.pillow_scaler import PillowScaler, Algorithm
from scale.attack import Attack


if __name__ == '__main__':

    data_transform = {
        'train':   transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((112, 112), interpolation=Image.NEAREST),
                transforms.ToTensor()

            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize((112, 112), interpolation=Image.NEAREST),
                transforms.ToTensor()

            ]
        )
    }

    load_train = get_loader(r'/home/pub-60/benign/train', data_transform['train'], batch=75)
    load_test = get_loader(r'/home/pub-60/benign/val', data_transform['test'], batch=100, shuffle=False)
    load_class_tar = get_loader(r'/home/pub-60/benign/val_class_1', data_transform['test'], batch=100, shuffle=False)
    load_backdoor_train = get_loader('/home/pub-60/backdoor/train', data_transform['train'], batch=75)
    load_backdoor_test = get_loader(r'/home/pub-60/backdoor/test', data_transform['test'], batch=100, shuffle=False)
    load_scale = get_loader(r'/home/pub-60/scale', data_transform['test'], batch=100, shuffle=False)

    torch.cuda.empty_cache()
    model = models.get_model('resnet18', 60)
    trainer = Trainer(100, model, best_acc=0.8, save_model='./resnet_pub.pth')
    # trainer.train_benign(load_train, load_test)
    trainer.train_backdoor(load_backdoor_train, load_test, load_backdoor_test, load_class_tar, load_scale,
                           './resnet_tiny.csv', './resnet-tiny.csv')

