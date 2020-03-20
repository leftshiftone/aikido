from dataclasses import dataclass, field
from typing import List


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
    tv_weight = 1e-3
    num_iterations: int = 1000
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
    content_layers: str = "relu4_2"
    style_layers: str = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'
    multidevice_strategy: str = '4,7,29'
