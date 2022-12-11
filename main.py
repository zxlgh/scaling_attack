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

    # load_train = get_loader(r'/home/tiny-image/benign/train', data_transform['train'], batch=256)
    # load_test = get_loader(r'/home/tiny-image/benign/val', data_transform['test'], batch=100, shuffle=False)
    # load_backdoor_train = get_loader('/home/tiny-image/backdoor/train', data_transform['train'], batch=256)
    # load_backdoor_test = get_loader(r'/home/tiny-image/backdoor/test', data_transform['test'], batch=100, shuffle=False)

    load_train = get_loader(r'/home/pub-60/benign/train', data_transform['train'], batch=256)
    load_test = get_loader(r'/home/pub-60/benign/val', data_transform['test'], batch=100, shuffle=False)
    load_backdoor_train = get_loader('/home/pub-60/backdoor/train', data_transform['train'], batch=256)
    load_backdoor_test = get_loader(r'/home/pub-60/backdoor/test', data_transform['test'], batch=100, shuffle=False)

    acc_train = []
    acc_test = []
    asr = []
    torch.cuda.empty_cache()
    model = models.get_model('resnet18', 60)
    trainer = Trainer(model)
    trainer.model.load_state_dict(torch.load('./resnet_pub.pth'))
    for e in range(100):
        print(f'Epoch: {e}')
        loss, acc = trainer.train(load_backdoor_train)
        acc_train.append(acc)
        print(f'train loss: {loss:.2f}\t acc: {acc:.2f}')
        loss, acc = trainer.test(load_test)
        acc_test.append(acc)
        print(f'test loss: {loss:.2f}\t acc: {acc:.2f}')
        _, acc = trainer.test(load_backdoor_test)
        asr.append(acc)
        print(f'asr: {acc:.2f}')




