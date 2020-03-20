import os
from dataclasses import dataclass

from torch.nn import Parameter

from aikido.aikidoka.styletransfer import StyleTransferKun
from aikido.nn.modules.styletransfer.preprocessing import deprocess


@dataclass
class CheckpointListener:
    kun: StyleTransferKun

    def maybe_print(self, t, loss, content_losses, style_losses):
        if self.kun.print_iter > 0 and t % self.kun.print_iter == 0:
            print("Iteration " + str(t) + " / " + str(self.kun.num_iterations))
            for i, loss_module in enumerate(content_losses):
                print("  Content " + str(i + 1) + " loss: " + str(loss_module.loss.item()))
            for i, loss_module in enumerate(style_losses):
                print("  Style " + str(i + 1) + " loss: " + str(loss_module.loss.item()))
            print("  Total loss: " + str(loss.item()))

    def maybe_save(self, t, img:Parameter, content_image):
        should_save = self.kun.save_iter > 0 and t % self.kun.save_iter == 0
        should_save = should_save or t == self.kun.num_iterations
        if should_save:
            output_filename, file_extension = os.path.splitext(self.kun.output_image)
            if t == self.kun.num_iterations:
                filename = output_filename + str(file_extension)
            else:
                filename = str(output_filename) + "_" + str(t) + str(file_extension)
            disp = deprocess(img.clone())

            # Maybe perform postprocessing for color-independent style transfer
            if self.kun.original_colors == 1:
                disp = self.original_colors(deprocess(content_image.clone()), disp)

            disp.save(str(filename))
