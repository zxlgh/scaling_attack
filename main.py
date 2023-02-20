import time

from torchvision import transforms
from PIL import Image
import torch
import models
import utils.plot
from train_test import Trainer
from datasets import get_loader


if __name__ == '__main__':

    data_transform = {
        'train':   transforms.Compose(
            [
                # transforms.RandomCrop(256, padding=4),
                # transforms.RandomRotation(90),
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()

            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=Image.NEAREST),
                transforms.ToTensor()

            ]
        )
    }

    load_train = get_loader(r'/home/pub-60/attack/train', data_transform['train'], batch=256)
    load_test = get_loader(r'/home/pub-60/attack/val', data_transform['test'], batch=100, shuffle=False)
    # load_backdoor_train = get_loader('/home/pub-60/backdoor/train', data_transform['train'], batch=256)
    load_backdoor_test = get_loader(r'/home/pub-60/attack/val_attack', data_transform['test'], batch=100, shuffle=False)

    acc_train = []
    acc_test = []
    asr = []
    epoch = []
    torch.cuda.empty_cache()
    model = models.get_model('resnet18', 60)
    trainer = Trainer(model)
    # trainer.model.load_state_dict(torch.load('./resnet_pub.pth'))
    for e in range(100):

        print(f'Epoch: {e}')
        epoch.append(e)
        time.sleep(0.01)
        loss, acc = trainer.train(load_train)
        print(f'train loss: {loss:.2f}\t acc: {acc:.2f}')
        time.sleep(0.01)
        loss, acc = trainer.test(load_test)
        acc_test.append(acc)
        print(f'test loss: {loss:.2f}\t acc: {acc:.2f}')
        time.sleep(0.01)
        loss, acc = trainer.test(load_backdoor_test)
        asr.append(acc)
        print(f'asr: {acc:.2f}')

    utils.plot.plot_train_process(epoch, {}, acc={'Accuracy': acc_test, 'ASR': asr}, name='/home/pub-60/attack/pubfig.eps')





