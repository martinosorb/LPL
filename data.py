from torchvision.datasets import MNIST
from torchvision import transforms
import torch

mnist_ds = MNIST('./data/', train=True, download=True,
                 transform=transforms.ToTensor())


def add_noise(img):
    noise = torch.randn(img.size(), device=img.device)*0.2
    img = torch.clamp(img+noise, 0., 1.)
    return img


noise_transform = transforms.Lambda(add_noise)
