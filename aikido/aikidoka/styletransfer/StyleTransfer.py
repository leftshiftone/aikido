import logging

import torch
import torch.nn as nn
from PIL import Image

from aikido.__api__.Aikidoka import Aikidoka
from aikido.aikidoka.styletransfer.ModelParallel import ModelParallel
from aikido.aikidoka.styletransfer.StyleTransferKun import StyleTransferKun
from aikido.aikidoka.styletransfer.VGG19 import loadVGG19
from aikido.nn.modules.styletransfer.ContentLoss import ContentLoss
from aikido.nn.modules.styletransfer.StyleLoss import StyleLoss
from aikido.nn.modules.styletransfer.TVLoss import TVLoss
from aikido.nn.modules.styletransfer.fileloader import preprocess

Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images


# https://arxiv.org/pdf/1605.04603.pdf
class StyleTransfer(Aikidoka):

    def __init__(self, kun: StyleTransferKun):
        super().__init__()

        cnn, layerList = loadVGG19(kun)

        content_image = preprocess(kun.content_image, kun.image_size, kun).type(kun.get_dtype())
        styling_image = preprocess(kun.styling_image, kun.image_size * kun.style_scale, kun).type(kun.get_dtype())

        content_losses, styling_losses, tv_losses = [], [], []
        net = nn.Sequential()

        if kun.tv_weight > 0:
            tv_mod = TVLoss().type(kun.get_dtype())
            net.add_module("tv_loss", tv_mod)
            tv_losses.append(tv_mod)

        conv, relu, pool, norm = 0, 0, 0, 0
        for layer in cnn.children():
            add_content_loss = False
            add_styling_loss = False

            if isinstance(layer, nn.Conv2d):
                add_content_loss = layerList['C'][conv] in kun.content_layers
                add_styling_loss = layerList['C'][conv] in kun.styling_layers
                name = layerList["C"][conv]
                conv += 1
            elif isinstance(layer, nn.ReLU):
                add_content_loss = layerList['R'][relu] in kun.content_layers
                add_styling_loss = layerList['R'][relu] in kun.styling_layers
                name = layerList["R"][relu]
                relu += 1
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                #layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
                name = layerList["P"][pool]
                pool += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(norm)
                norm += 1
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            net.add_module(name, layer)

            if add_content_loss:
                with torch.no_grad():
                    target = ModelParallel(net, kun)(content_image).detach()
                    content_loss = ContentLoss(target)
                    content_losses.append(content_loss)
                    net.add_module("content_loss_{}".format(len(content_losses)), content_loss)
                torch.cuda.empty_cache()

            if add_styling_loss:
                with torch.no_grad():
                    target_feature = ModelParallel(net, kun)(styling_image).detach()
                    styling_loss = StyleLoss(target_feature)
                    styling_losses.append(styling_loss)
                    net.add_module("style_loss_{}".format(len(styling_losses)), styling_loss)
                torch.cuda.empty_cache()

        # now we trim off the layers after the last content and style losses
        for i in range(len(net) - 1, -1, -1):
            if isinstance(net[i], ContentLoss) or isinstance(net[i], StyleLoss):
                net = net[:(i + 1)]
                break

        logging.info(net)

        for param in net.parameters():
            param.requires_grad = False

        if len(kun.gpu_sections) > 0:
            net = ModelParallel(net, kun)

        self.kun = kun
        self.net = net
        self.content_image = content_image
        self.styling_image = styling_image
        self.content_losses, self.styling_losses, self.tv_losses = content_losses, styling_losses, tv_losses

    def forward(self, x):
        return self.net(x)
