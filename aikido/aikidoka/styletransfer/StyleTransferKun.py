from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch


class Pooling(Enum):
    MAX = "max"
    AVG = "avg"

@dataclass
class StyleTransferKun:
    styling_image: str
    content_image: str
    image_size: int = 512
    # Zero-indexed ID of the GPU to use; empty list for CPU mode
    gpu: List[int] = field(default_factory=lambda: [0])
    tv_weight: float = 1e-3
    style_scale: float = 1.0
    caffee_model: bool = False
    model: str = None
    file_name:str = "output.png"
    pooling: Pooling = Pooling.MAX
    content_layers: List[str] = field(default_factory=lambda: ['conv4_1'])
    styling_layers: List[str] = field(default_factory=lambda: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
    content_weights: List[float] = field(default_factory=lambda: [5e0])
    styling_weights: List[float] = field(default_factory=lambda: [1e2, 1e2, 1e2, 1e2, 1e2])

    def get_backward_device(self):
        if len(self.gpu) > 1:
            if len(self.gpu) == 0:
                return "cpu"
            return "cuda:" + str(self.gpu[0])
        elif len(self.gpu) > 0:
            return "cuda:" + str(self.gpu[0])
        return "cpu"

    def get_dtype(self):
        if len(self.gpu) > 1:
            return torch.FloatTensor
        if len(self.gpu) > 0:
            return torch.cuda.FloatTensor
        return torch.FloatTensor
