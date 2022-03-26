import torch
import torch.nn as nn


class ModuleParser:
    supported_modules = {
        'Linear': nn.Linear,
        'Conv2d': nn.Conv2d,
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'BatchNorm2d': nn.BatchNorm2d,
        'MaxPool2d': nn.MaxPool2d,
        'AvgPool2d': nn.AvgPool2d,
        'AdaptiveMaxPool2d': nn.AdaptiveMaxPool2d,
        'AdaptiveAvgPool2d': nn.AdaptiveAvgPool2d,
        'Softmax': nn.Softmax
    }

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.modules = ModuleParser.parse_submodule(model)
        self.record = None
        self.record_submodule()

    @staticmethod
    def parse_submodule(model: torch.nn.Module):
        modules = []
        for name, subm in model.named_children():
            if len(subm._modules) > 0:
                mod = ModuleParser.parse_submodule(subm)
                mod.insert(0, name)
            else:
                mod = [name, subm]
            modules.append(mod)
        return modules

    def record_submodule(self):

        pass

