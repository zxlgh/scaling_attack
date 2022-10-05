import torch
import torch.nn as nn
from torchvision.models import resnet, vgg


def get_model(model_name, out_feature, load_dict=None):
    """

    :param model_name:
    :param out_feature: The number of classes
    :param load_dict:   The address of state_dict.
    :return:
    """
    if model_name == 'resnet18':
        model = resnet.resnet18(pretrained=True)
        model.fc = nn.Linear(512, out_feature)
    elif model_name == 'vgg16':
        model = vgg.vgg16(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_feature),
        )
    else:
        raise Exception('Model_name is wrong.')
    if load_dict is not None:
        model.load_state_dict(torch.load(load_dict))
    return model
