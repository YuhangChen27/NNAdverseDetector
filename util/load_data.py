import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

in_built = ['CIFAR10', 'CIFAR100', 'Fashion-MNIST', 'GTSRB', 'ImageNet', 'LSUN', 'MNIST', 'Omniglot', 'STL-10', 'SVHN']


class SelectDataset:
    built_in = {'CIFAR10': torchvision.datasets.CIFAR10,
                'CIFAR100': torchvision.datasets.CIFAR100,
                'Fashion-MNIST': torchvision.datasets.FashionMNIST,
                # 'GTSRB': torchvision.datasets.GTSRB,
                'ImageNet': torchvision.datasets.ImageNet,
                'LSUN': torchvision.datasets.LSUN,
                'MNIST': torchvision.datasets.MNIST,
                'Omniglot': torchvision.datasets.Omniglot,
                'Places365': torchvision.datasets.Places365,
                'STL10': torchvision.datasets.STL10,
                'SVHN': torchvision.datasets.SVHN}

    def __init__(self, name: str, root, transform, **kwargs):
        self.dataset = None
        if name not in self.built_in:
            raise Exception('Please Customize Dataset')
        else:
            if root is None:
                self.dataset = self.built_in[name](root=f'../data/{name}', transform=transform, download=True, **kwargs)
            else:
                self.dataset = self.built_in[name](root=root, transform=transform, **kwargs)


if __name__ == '__main__':
    mnist = SelectDataset(name='STL10', root=None,
                          transform=transforms.Compose([
                                                        # transforms.ToTensor(),
                                                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                        transforms.Resize(224)
                                                                              ]))
    mnist_dataset = mnist.dataset
    print()
