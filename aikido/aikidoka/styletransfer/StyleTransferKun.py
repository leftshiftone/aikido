from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class StyleTransferKun:
    style_image: str
    content_image: str
    style_blend_weights = None
    image_size: int = 512
    # Zero-indexed ID of the GPU to use; empty list for CPU mode
    gpu: List[int] = field(default_factory=lambda: [0])
    content_weight: float = 2e0
    style_weight: float = 5e2
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
    pooling: str = "max"  # avg, max
    model_file: str = 'D:/IntellijProjects/aikido/aikido/aikidoka/styletransfer/models/vgg19-d01eb7cb.pth'
    disable_check: bool = False
    backend: str = "nn"  # 'nn', 'cudnn', 'mkl', 'mkldnn', 'openmp', 'mkl,cudnn', 'cudnn,mkl'
    cudnn_autotune: bool = False
    seed: int = -1
    content_layers: List[str] = field(default_factory=lambda: ['relu4_2'])
    style_layers: List[str] = field(default_factory=lambda: ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    multidevice_strategy: str = '4,7,29'

    def get_style_blend_weights(self, style_image_list):
        # Handle style blending weights for multiple style inputs
        style_blend_weights = []
        if self.style_blend_weights == None:
            # Style blending not specified, so use equal weighting
            for i in style_image_list:
                style_blend_weights.append(1.0)
            for i, blend_weights in enumerate(style_blend_weights):
                style_blend_weights[i] = int(style_blend_weights[i])
        else:
            style_blend_weights = self.style_blend_weights.split(',')
            assert len(style_blend_weights) == len(style_image_list), \
                "-style_blend_weights and -style_images must have the same number of elements!"

        # Normalize the style blending weights so they sum to 1
        style_blend_sum = 0
        for i, blend_weights in enumerate(style_blend_weights):
            style_blend_weights[i] = float(style_blend_weights[i])
            style_blend_sum = float(style_blend_sum) + style_blend_weights[i]
        for i, blend_weights in enumerate(style_blend_weights):
            style_blend_weights[i] = float(style_blend_weights[i]) / float(style_blend_sum)

        return style_blend_weights

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

    def is_multidevice(self):
        return len(self.gpu) > 1
