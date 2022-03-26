import torch


class LinearHook:
    def __init__(self, module):
        self.weights = module.weight.data
        self.bias = module.bias.data
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input[0].data
        self.output = output.data

    def calc(self):
        self.noact_output = torch.matmul(self.input, self.weights.T) + self.bias

    def close(self):
        self.hook.remove()
