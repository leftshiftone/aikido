import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from aikido.__api__.Aikidoka import Aikidoka
from aikido.aikidoka.styletransfer.listener.CheckpointListener import CheckpointListener
from aikido.dojo.listener import SeedListener, BackendListener

Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images

from aikido.aikidoka.styletransfer.VGG19 import loadVGG19

from aikido.aikidoka.styletransfer.StyleTransferKun import StyleTransferKun
from aikido.nn.modules.styletransfer.ContentLoss import ContentLoss
from aikido.nn.modules.styletransfer.StyleLoss import StyleLoss
from aikido.nn.modules.styletransfer.TVLoss import TVLoss
# from aikido.nn.modules.styletransfer.preprocessing import preprocess
from aikido.nn.modules.styletransfer.fileloader import preprocess
from aikido.nn.modules.styletransfer.Normalization import Normalization


# https://arxiv.org/pdf/1605.04603.pdf
class StyleTransfer(Aikidoka):

    def __init__(self, kun: StyleTransferKun):
        super().__init__()

        cnn, layerList = loadVGG19(kun)

        content_image = preprocess(kun.content_image, kun.image_size, kun).type(kun.get_dtype())
        styling_image = preprocess(kun.styling_image, kun.image_size, kun).type(kun.get_dtype())

        # assert styling_image.size() == content_image.size(), "content/styling image size mismatch"

        # Set up the network, inserting style and content loss modules
        content_losses, styling_losses, tv_losses = [], [], []
        net = nn.Sequential(Normalization().to(get_device()))

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
                name = layerList["P"][pool]
                pool += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(norm)
                norm += 1
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            net.add_module(name, layer)

            if add_content_loss:
                target = net(content_image).detach()
                content_loss = ContentLoss(target)
                content_losses.append(content_loss)
                net.add_module("content_loss_{}".format(len(content_losses)), content_loss)

            if add_styling_loss:
                target_feature = net(styling_image).detach()
                styling_loss = StyleLoss(target_feature)
                styling_losses.append(styling_loss)
                net.add_module("style_loss_{}".format(len(styling_losses)), styling_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(net) - 1, -1, -1):
            if isinstance(net[i], ContentLoss) or isinstance(net[i], StyleLoss):
                net = net[:(i + 1)]
                break

        print(net)

        for param in net.parameters():
            param.requires_grad = False

        self.kun = kun
        self.net = net
        self.content_image = content_image
        self.styling_image = styling_image
        self.content_losses, self.styling_losses, self.tv_losses = content_losses, styling_losses, tv_losses

    def forward(self):
        BackendListener(self.kun.backend, self.kun.cudnn_autotune).training_started(None, None, None)
        SeedListener(self.kun.seed).training_started(None, None, None)

        generated_image = self.init_image()
        generated_image = nn.Parameter(generated_image)
        optimizer = self.setup_optimizer(generated_image, self.kun)

        # Function to evaluate loss and gradient. We run the net forward and
        # backward to get the gradient, and sum up losses from the loss modules.
        # optim.lbfgs internally handles iteration and calls this function many
        # times, so we manually count the number of iterations to handle printing
        # and saving intermediate results.
        dan = [0]

        while dan[0] <= self.kun.dans:
            def closure():
                optimizer.zero_grad()
                self.net(generated_image)

                content_score, styling_score, tv_score = 0, 0, 0

                for sl in self.styling_losses:
                    styling_score += sl.loss.to(self.kun.get_backward_device())
                for cl in self.content_losses:
                    content_score += cl.loss.to(self.kun.get_backward_device())
                for tl in self.tv_losses:
                    tv_score += tl.loss.to(self.kun.get_backward_device())

                styling_score *= self.kun.styling_weight
                content_score *= self.kun.content_weight
                tv_score *= self.kun.tv_weight

                loss = styling_score + content_score + tv_score
                loss.backward()

                listener = CheckpointListener(self.kun)

                listener.maybe_save(dan[0], generated_image, self.content_image)
                listener.maybe_print(dan[0], loss, self.content_losses, self.styling_losses, self.tv_losses)

                dan[0] += 1
                return styling_score + content_score + tv_score

            optimizer.step(closure)

        # generated_image.data.clamp_(0, 1)
        return generated_image

    # Configure the optimizer
    def setup_optimizer(self, img, kun):
        print("Running optimization with L-BFGS")
        optim_state = {
            'history_size': kun.lbfgs_num_correction
        }
        return optim.LBFGS([img.requires_grad_()], **optim_state)

    def init_image(self):
        if self.kun.init == 'random':
            B, C, H, W = self.content_image.size()
            factor = 0.001 if self.kun.caffee_model else 1

            return torch.randn(C, H, W).mul(factor).unsqueeze(0).type(self.kun.get_dtype())
        elif self.kun.init == 'image':
            if self.kun.init_image != None:
                image_size = (self.content_image.size(2), self.content_image.size(3))
                init_image = preprocess(self.kun.init_image, image_size).type(self.kun.get_dtype())
                return init_image.clone()
            else:
                return self.content_image.clone()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
