from torchvision import transforms
import torch


def add_noise(img):
    noise = torch.randn(img.size(), device=img.device)*0.2
    img = torch.clamp(img+noise, 0., 1.)
    return img


noise_transform = transforms.Lambda(add_noise)

multiple_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2)),
])
