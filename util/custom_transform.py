import torch
import torchvision
import torchvision.transforms.functional as TF


class Translation2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, img: torch.Tensor):
        return TF.affine(img, angle=0, translate=[self.x, self.y], scale=1.0,
                         interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=[0.0], shear=[0.0])
