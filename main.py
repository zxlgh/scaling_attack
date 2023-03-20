import time

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch

from utils.utils import set_seed
from utils.utils import get_model
from utils.trainHelper import Trainer
from utils.utils import get_loader


if __name__ == '__main__':

    set_seed(1234)

    data_transform = {
        'train':   transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),

            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor()

            ]
        )
    }

    load_train = get_loader(r'/opt/data/private/pub-60/train-camouflage', data_transform['train'], batch=200)
    load_test = get_loader(r'/opt/data/private/pub-60/val-original', data_transform['test'], batch=100, shuffle=True)
    load_backdoor_test = get_loader(r'/opt/data/private/pub-60/val-camouflage', data_transform['test'], batch=100, shuffle=False)

    torch.cuda.empty_cache()
    model = get_model('resnet18', 60)
    trainer = Trainer(model)

    best_cda = 0.0
    best_asr = 0.0
    for e in range(100):
        print(f'Epoch: {e}')
        time.sleep(0.01)
        loss, acc = trainer.train(load_train)
        print(f'train loss: {loss:.4f}\t acc: {acc:.4f}')
        time.sleep(0.01)
        loss, acc = trainer.test(load_test)
        if acc > best_cda:
            best_cda = acc
        print(f'test loss: {loss:.4f}\t acc: {acc:.4f}')
        time.sleep(0.01)
        _, asr = trainer.test(load_backdoor_test)
        if asr > best_asr:
            best_asr = asr
        print(f'asr: {asr:.4f}')
        time.sleep(0.01)

    print(best_cda)
    print(best_asr)
