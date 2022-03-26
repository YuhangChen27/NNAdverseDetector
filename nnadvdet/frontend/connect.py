import torch
from nnadvdet.core.detector import AdverseDetector
from nnadvdet.frontend.module_parse import ModuleParser


class Connector:
    def __init__(self, model: torch.nn.Module, config):
        self.parser = ModuleParser(model)
        self.detector = AdverseDetector(config)


