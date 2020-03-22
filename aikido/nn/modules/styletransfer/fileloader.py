import torch
import torchvision.transforms as transforms
from PIL import Image

from aikido.aikidoka.styletransfer import StyleTransferKun


def preprocess(image_name, image_size, kun: StyleTransferKun):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    if kun.caffee_model:
        Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
        rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    else:
        Normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1,1,1])])
        tensor = Normalize(Loader(image)).unsqueeze(0)

    return tensor


#  Undo the above preprocessing.
def deprocess(output_tensor, kun: StyleTransferKun):
    if kun.caffee_model:
        bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
        Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
        output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 256
    else:
        Normalize = transforms.Compose([transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1,1,1])])
        output_tensor = Normalize(output_tensor.squeeze(0).cpu())
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image
