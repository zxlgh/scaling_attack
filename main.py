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

    load_train = get_loader(r'/home/tiny-image/benign/train', data_transform['train'], batch=256)
    load_test = get_loader(r'/home/tiny-image/benign/val', data_transform['test'], batch=100, shuffle=False)
    load_backdoor_train = get_loader('/home/tiny-image/backdoor/train', data_transform['train'], batch=256)
    load_backdoor_test = get_loader(r'/home/tiny-image/backdoor/test', data_transform['test'], batch=100, shuffle=False)

    torch.cuda.empty_cache()
    model = models.get_model('resnet18', 200)
    trainer = Trainer(model, best_acc=0.6, save_model='./resnet_tiny.pth')



