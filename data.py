from torchvision import transforms
import torch


def add_noise(img):
    noise = torch.randn(img.size(), device=img.device)*0.2
    img = torch.clamp(img+noise, 0., 1.)
    return img


noise_transform = transforms.Lambda(add_noise)

multiple_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
    transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2)),
])
