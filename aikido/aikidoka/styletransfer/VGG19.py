import copy

import torch
import torchvision.models as models

from aikido.aikidoka.styletransfer import StyleTransferKun

vgg19_dict = {
    'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
          'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
    'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2',
          'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
    'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}


# Load the model, and configure pooling layer type
def loadVGG19(kun: StyleTransferKun):
    cnn = models.vgg19(pretrained=True).to(get_device()).eval()
    if kun.caffee_model:
        cnn.load_state_dict(torch.load("D:/IntellijProjects/aikido/aikido/aikidoka/styletransfer/models/vgg19-d01eb7cb.pth"), strict=(not False))

    cnn = cnn.features
    cnn = copy.deepcopy(cnn)
    #for param in cnn.parameters():
    #    param.requires_grad = False

    cnn = cnn.cuda()

    return cnn, vgg19_dict


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
