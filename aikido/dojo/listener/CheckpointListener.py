import os
from dataclasses import dataclass

from torch.nn import Parameter

from aikido.__api__ import DojoListener, Aikidoka


@dataclass
class CheckpointListener(DojoListener):
    print_iter:int = 5
    save_iter:int = 5
    original_colors: bool = False

    def dan_finished(self, aikidoka: Aikidoka, run: (int, int), metrics: (float, float)):
        self.kun = aikidoka.kun
        self.maybe_print(run[0] + 1, metrics[0], aikidoka.content_losses, aikidoka.styling_losses, aikidoka.tv_losses)
        self.maybe_save(run[0] + 1, aikidoka.generated_image, aikidoka.content_image)

    def maybe_print(self, t, loss, content_losses, styling_losses, tv_losses):
        if self.print_iter > 0 and t % self.print_iter == 0:
            print("Iteration " + str(t) + " / " + str(50))# FIXME
            for i, loss_module in enumerate(content_losses):
                print("  Content " + str(i + 1) + " loss: " + str(loss_module.loss.item() * self.kun.content_weights[i]))
            for i, loss_module in enumerate(styling_losses):
                print("  Style " + str(i + 1) + " loss: " + str(loss_module.loss.item() * self.kun.styling_weights[i]))
            for i, loss_module in enumerate(tv_losses):
                print("  TV " + str(i + 1) + " loss: " + str(loss_module.loss.item() * self.kun.tv_weight))
            print("  Total loss: " + str(loss.item()))

    def maybe_save(self, t, img: Parameter, content_image):
        from aikido.nn.modules.styletransfer.fileloader import deprocess

        should_save = self.save_iter > 0 and t % self.save_iter == 0 and t > 0
        should_save = should_save or t == 50#FIXME
        if should_save:
            output_filename, file_extension = os.path.splitext(self.kun.file_name)
            filename = str(output_filename) + "_" + str(t) + str(file_extension)

            # disp = deprocess(img.squeeze(0).clone())
            disp = deprocess(img.clone(), self.kun)

            # Maybe perform postprocessing for color-independent style transfer
            if self.original_colors:
                disp = self.postprocess_colors(deprocess(content_image.clone(), self.kun), disp)

            disp.save(str(filename))

    # Combine the Y channel of the generated image and the UV/CbCr channels of the
    # content image to perform color-independent style transfer.
    def postprocess_colors(self, content, generated):
        from PIL import Image
        content_channels = list(content.convert('YCbCr').split())
        generated_channels = list(generated.convert('YCbCr').split())
        content_channels[0] = generated_channels[0]
        return Image.merge('YCbCr', content_channels).convert('RGB')
