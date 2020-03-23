import logging

import torch
import torch.nn as nn
import torch.optim as optim

from aikido.__api__ import Aikidoka, DojoKun, Kata
from aikido.dojo import BaseDojo
from aikido.dojo.listener.CheckpointListener import CheckpointListener
from aikido.nn.modules.styletransfer.fileloader import preprocess


class StyleTransferDojoKun(DojoKun):
    def __init__(self, dans=50, init_image: str = None, init: str = "random",
                 lbfgs_num_correction: int = 100, geometric_weight: bool = False
                 , original_colors: bool = False):
        super().__init__(None, None, dans)
        self.init = init
        self.init_image = init_image
        self.lbfgs_num_correction = lbfgs_num_correction
        self.geometric_weight = geometric_weight
        self.original_colors = original_colors


class StyleTransferDojo(BaseDojo):

    def __init__(self, dojokun: StyleTransferDojoKun):
        super().__init__(dojokun)
        self.add_listener(CheckpointListener(original_colors=dojokun.original_colors))

    def _before_training_started(self, aikidoka: Aikidoka):
        self.content_image = aikidoka.content_image
        self.kun = aikidoka.kun
        self.generated_image = self._init_image()
        self.generated_image = nn.Parameter(self.generated_image)
        aikidoka.generated_image = self.generated_image
        self.optimizer = self._setup_optimizer(self.generated_image, self.dojokun)

    # Function to evaluate loss and gradient. We run the net forward and
    # backward to get the gradient, and sum up losses from the loss modules.
    # optim.lbfgs internally handles iteration and calls this function many
    # times, so we manually count the number of iterations to handle printing
    # and saving intermediate results.
    def _do_dan(self, aikidoka: Aikidoka, kata: Kata):

        loss_ref = [0]

        def closure():
            self.optimizer.zero_grad()
            aikidoka(self.generated_image)

            loss = 0
            for i, sl in enumerate(aikidoka.styling_losses):
                weight = self.kun.styling_weight / (2 ** i if self.dojokun.geometric_weight else 1)
                loss += sl.loss.to(self.kun.get_backward_device()) * weight
            for i, cl in enumerate(aikidoka.content_losses):
                weight = self.kun.content_weight / (
                    2 ** (len(aikidoka.content_losses) - i) if self.dojokun.geometric_weight else 1)
                loss += cl.loss.to(self.kun.get_backward_device()) * weight
            for tl in aikidoka.tv_losses:
                loss += tl.loss.to(self.kun.get_backward_device()) * self.kun.tv_weight

            loss.backward()

            loss_ref[0] = loss
            return loss

        self.optimizer.step(closure)
        return loss_ref[0], 0

    def _setup_optimizer(self, img, kun):
        optim_state = {'history_size': kun.lbfgs_num_correction}
        return optim.LBFGS([img.requires_grad_()], **optim_state)

    def _init_image(self):
        if self.dojokun.init == 'random':
            logging.info("random initialization")
            b, c, h, w = self.content_image.size()
            return torch.randn(c, h, w).mul(0.001).unsqueeze(0).type(self.kun.get_dtype())
        elif self.dojokun.init == 'image':
            if self.dojokun.init_image is not None:
                logging.info("image initialization")
                image_size = (self.content_image.size(2), self.content_image.size(3))
                init_image = preprocess(self.dojokun.init_image, image_size, self.kun).type(
                    self.kun.get_dtype())
                return init_image.clone()
        else:
            logging.info("content initialization")
            return self.content_image.clone()
