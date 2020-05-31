from typing import List

import torch
from torch import nn

from aikido.aikidoka.styletransfer import StyleTransferKun


class ModelParallel(nn.Module):
    """
    This model is used to support multi gpu instances.
    """

    def __init__(self, net: nn.Module, kun: StyleTransferKun):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(kun.gpu)
        self.sections = self.section_to_devices(self.get_sections(net, kun.gpu_sections.copy()))

        # print("devices")
        # print(self.device_list)
        # print("sections")
        # print(self.sections)

    """
    Translates each gpu index to a device name.
    When no gpu is used a single cpu device is returned.
    """
    def name_devices(self, gpu_list:List[int]):
        if len(gpu_list) == 0:
            return ["cpu"]
        return list(map(lambda x: "cuda:" + str(x), gpu_list))

    """
    Splits the model to sections for each gpu/cpu instance.
    """
    def get_sections(self, net: nn.Module, gpu_sections: List[int]):
        sections, cur_section = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_section.add_module(str(i), net[i])
            if i in gpu_sections:
                del gpu_sections[0]
                sections.append(cur_section)
                cur_section = nn.Sequential()
        sections.append(cur_section)
        return sections

    """
    Maps each section to one or more device(s).
    """
    def section_to_devices(self, sections):
        for i, section in enumerate(sections):
            section.to(self.device_list[i])
        return sections

    def c(self, x, i: int):
        if x.type() == 'torch.FloatTensor' and 'cuda' in self.device_list[i]:
            x = x.type('torch.cuda.FloatTensor')
        elif x.type() == 'torch.cuda.FloatTensor' and 'cpu' in self.device_list[i]:
            x = x.type('torch.FloatTensor')
        return x

    def forward(self, x):
        for i, section in enumerate(self.sections):
            if i < len(self.sections) -1:
                device_1 = self.device_list[i]
                device_2 = self.device_list[i + 1]

                x = section(self.c(x, i).to(device_1))
                x = self.c(x, i + 1).to(device_2)
            else:
                x = section(x)

        return x
