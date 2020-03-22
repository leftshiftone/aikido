from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class StyleTransferKun:
    styling_image: str
    content_image: str
    style_blend_weights = None
    image_size: int = 512
    # Zero-indexed ID of the GPU to use; empty list for CPU mode
    gpu: List[int] = field(default_factory=lambda: [0])
    content_weight: float = 5e0
    styling_weight: float = 1e2
    normalize_weights: bool = False
    tv_weight:float = 1e-3
    dans: int = 50
    init: str = "random"  # random, image
    init_image = None
    lbfgs_num_correction: int = 100
    print_iter: int = 50
    save_iter: int = 100
    output_image: str = "out.png"
    style_scale: float = 1.0
    original_colors: int = 0  # 0, 1
    caffee_model: bool = False
    geometric_weight:bool = False
    pooling: str = "max"  # avg, max
    backend: str = "nn"  # 'nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'
    cudnn_autotune: bool = False
    seed: int = -1
    content_layers: List[str] = field(default_factory=lambda: ['conv4_1'])
    #content_layers: List[str] = field(default_factory=lambda: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'])
    styling_layers: List[str] = field(default_factory=lambda: ['conv1_1','conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
    #styling_layers: List[str] = field(default_factory=lambda: ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'])
    multidevice_strategy: str = '4,7,29'

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
