import torch
import torchvision.transforms as transforms

from PIL import Image


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size)) * x) for x in (image.height, image.width)])

    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()])

    image = loader(image).unsqueeze(0)
    return image.to(get_device(), torch.float)


#  Undo the above preprocessing.
def deprocess(output_tensor):
    output_tensor = output_tensor.cpu().clone()
    output_tensor = output_tensor.squeeze(0)

    Image2PIL = transforms.ToPILImage()
    return Image2PIL(output_tensor)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
