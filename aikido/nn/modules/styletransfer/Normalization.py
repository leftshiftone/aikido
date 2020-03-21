import torch
import torch.nn as nn


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(get_device())
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(get_device())


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(cnn_normalization_mean).view(-1, 1, 1)
        self.std = torch.tensor(cnn_normalization_std).view(-1, 1, 1)

    def forward(self, img):
        #return (img - self.mean) / self.std#

        return img
