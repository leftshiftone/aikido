import torch.nn as nn
import torch.nn.functional as F


# Define an nn Module to compute content loss
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        if (input.size() == self.target.size()):
            self.loss = F.mse_loss(input, self.target)
        return input
