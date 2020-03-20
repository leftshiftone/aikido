import copy

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from aikido.__api__.Aikidoka import Aikidoka
from aikido.aikidoka.styletransfer.listener.CheckpointListener import CheckpointListener
from aikido.dojo.listener import SeedListener, BackendListener

Image.MAX_IMAGE_PIXELS = 1000000000  # Support gigapixel images

from aikido.aikidoka.styletransfer.CaffeLoader import loadCaffemodel, ModelParallel

from aikido.aikidoka.styletransfer.StyleTransferKun import StyleTransferKun
from aikido.nn.modules.styletransfer.ContentLoss import ContentLoss
from aikido.nn.modules.styletransfer.StyleLoss import StyleLoss
from aikido.nn.modules.styletransfer.TVLoss import TVLoss
from aikido.nn.modules.styletransfer.preprocessing import preprocess
from aikido.__common__.io.Files import get_files


class StyleTransfer(Aikidoka):

    def __init__(self, kun: StyleTransferKun):
        super().__init__()
        self.kun = kun

    def forward(self):
        BackendListener(self.kun.backend, self.kun.cudnn_autotune).training_started(None, None, None)

        cnn, layerList = loadCaffemodel(self.kun.model_file, self.kun.pooling, self.kun.gpu, self.kun.disable_check)

        content_image = preprocess(self.kun.content_image, self.kun.image_size).type(self.kun.get_dtype())
        style_image_list = get_files(self.kun.style_image.split(','))

        style_images_caffe = []
        for image in style_image_list:
            style_size = int(self.kun.image_size * self.kun.style_scale)
            img_caffe = preprocess(image, style_size).type(self.kun.get_dtype())
            style_images_caffe.append(img_caffe)

        # Handle style blending weights for multiple style inputs
        style_blend_weights = self.kun.get_style_blend_weights(style_image_list)

        # Set up the network, inserting style and content loss modules
        cnn = copy.deepcopy(cnn)
        content_losses, style_losses, tv_losses = [], [], []
        next_content_idx, next_style_idx = 1, 1
        net = nn.Sequential()
        c, r = 0, 0
        if self.kun.tv_weight > 0:
            tv_mod = TVLoss(self.kun.tv_weight).type(self.kun.get_dtype())
            net.add_module(str(len(net)), tv_mod)
            tv_losses.append(tv_mod)

        for i, layer in enumerate(list(cnn), 1):
            if next_content_idx <= len(self.kun.content_layers) or next_style_idx <= len(self.kun.style_layers):
                if isinstance(layer, nn.Conv2d):
                    net.add_module(str(len(net)), layer)

                    if layerList['C'][c] in self.kun.content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = ContentLoss(self.kun.content_weight)
                        net.add_module(str(len(net)), loss_module)
                        content_losses.append(loss_module)

                    if layerList['C'][c] in self.kun.style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = StyleLoss(self.kun.style_weight)
                        net.add_module(str(len(net)), loss_module)
                        style_losses.append(loss_module)
                    c += 1

                if isinstance(layer, nn.ReLU):
                    net.add_module(str(len(net)), layer)

                    if layerList['R'][r] in self.kun.content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(layerList['R'][r]))
                        loss_module = ContentLoss(self.kun.content_weight)
                        net.add_module(str(len(net)), loss_module)
                        content_losses.append(loss_module)
                        next_content_idx += 1

                    if layerList['R'][r] in self.kun.style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(layerList['R'][r]))
                        loss_module = StyleLoss(self.kun.style_weight)
                        net.add_module(str(len(net)), loss_module)
                        style_losses.append(loss_module)
                        next_style_idx += 1
                    r += 1

                if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                    net.add_module(str(len(net)), layer)

        if self.kun.is_multidevice():
            net = self.setup_multi_device(net, self.kun)

        # Capture content targets
        for i in content_losses:
            i.mode = 'capture'
        print("Capturing content targets")
        self.print_torch(net)
        net(content_image)

        # Capture style targets
        for i in content_losses:
            i.mode = 'None'

        for i, image in enumerate(style_images_caffe):
            print("Capturing style target " + str(i + 1))
            for j in style_losses:
                j.mode = 'capture'
                j.blend_weight = style_blend_weights[i]
            net(style_images_caffe[i])

        # Set all loss modules to loss mode
        for i in content_losses:
            i.mode = 'loss'
        for i in style_losses:
            i.mode = 'loss'

        # Maybe normalize content and style weights
        if self.kun.normalize_weights:
            normalize_weights(content_losses, style_losses)

        # Freeze the network in order to prevent
        # unnecessary gradient calculations
        for param in net.parameters():
            param.requires_grad = False

        # Initialize the image
        SeedListener(self.kun.seed).training_started(None, None, None)
        img = nn.Parameter(self.init_image(content_image))

        # Function to evaluate loss and gradient. We run the net forward and
        # backward to get the gradient, and sum up losses from the loss modules.
        # optim.lbfgs internally handles iteration and calls this function many
        # times, so we manually count the number of iterations to handle printing
        # and saving intermediate results.
        num_calls = [0]

        def feval():
            num_calls[0] += 1
            optimizer.zero_grad()
            net(img)
            loss = 0

            for mod in content_losses:
                loss += mod.loss.to(self.kun.get_backward_device())
            for mod in style_losses:
                loss += mod.loss.to(self.kun.get_backward_device())
            if self.kun.tv_weight > 0:
                for mod in tv_losses:
                    loss += mod.loss.to(self.kun.get_backward_device())

            loss.backward()

            listener = CheckpointListener(self.kun)

            listener.maybe_save(num_calls[0], img, content_image)
            listener.maybe_print(num_calls[0], loss, content_losses, style_losses)

            return loss

        optimizer, loopVal = self.setup_optimizer(img, self.kun)
        while num_calls[0] <= loopVal:
            optimizer.step(feval)

    # Configure the optimizer
    def setup_optimizer(self, img, kun):
        print("Running optimization with L-BFGS")
        optim_state = {
            'max_iter': kun.num_iterations,
            'tolerance_change': -1,
            'tolerance_grad': -1,
        }
        if kun.lbfgs_num_correction != 100:
            optim_state['history_size'] = kun.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        return optimizer, 1

    def init_image(self, content_image):
        if self.kun.init == 'random':
            B, C, H, W = content_image.size()
            return torch.randn(C, H, W).mul(0.001).unsqueeze(0).type(self.kun.get_dtype())
        elif self.kun.init == 'image':
            if self.kun.init_image != None:
                image_size = (content_image.size(2), content_image.size(3))
                init_image = preprocess(self.kun.init_image, image_size).type(self.kun.get_dtype())
                return init_image.clone()
            else:
                return content_image.clone()

    def setup_multi_device(self, net, kun):
        assert len(kun.gpu.split(',')) - 1 == len(kun.multidevice_strategy.split(',')), \
            "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."

        new_net = ModelParallel(net, kun.gpu, kun.multidevice_strategy)
        return new_net

    # Combine the Y channel of the generated image and the UV/CbCr channels of the
    # content image to perform color-independent style transfer.
    def original_colors(self, content, generated):
        content_channels = list(content.convert('YCbCr').split())
        generated_channels = list(generated.convert('YCbCr').split())
        content_channels[0] = generated_channels[0]
        return Image.merge('YCbCr', content_channels).convert('RGB')

    # Print like Lua/Torch7
    def print_torch(self, net):
        if self.kun.is_multidevice():
            return
        simplelist = ""
        for i, layer in enumerate(net, 1):
            simplelist = simplelist + "(" + str(i) + ") -> "
        print("nn.Sequential ( \n  [input -> " + simplelist + "output]")

        def strip(x):
            return str(x).replace(", ", ',').replace("(", '').replace(")", '') + ", "

        def n():
            return "  (" + str(i) + "): " + "nn." + str(l).split("(", 1)[0]

        for i, l in enumerate(net, 1):
            if "2d" in str(l):
                ks, st, pd = strip(l.kernel_size), strip(l.stride), strip(l.padding)
                if "Conv2d" in str(l):
                    ch = str(l.in_channels) + " -> " + str(l.out_channels)
                    print(n() + "(" + ch + ", " + (ks).replace(",", 'x', 1) + st + pd.replace(", ", ')'))
                elif "Pool2d" in str(l):
                    st = st.replace("  ", ' ') + st.replace(", ", ')')
                    print(n() + "(" + ((ks).replace(",", 'x' + ks, 1) + st).replace(", ", ','))
            else:
                print(n())
        print(")")


# Divide weights by channel size
def normalize_weights(content_losses, style_losses):
    for n, i in enumerate(content_losses):
        i.strength = i.strength / max(i.target.size())
    for n, i in enumerate(style_losses):
        i.strength = i.strength / max(i.target.size())
