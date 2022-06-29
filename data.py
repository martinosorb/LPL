from torchvision import transforms as T
import torch


def add_noise(img):
    noise = torch.randn(img.size(), device=img.device)*0.2
    img = torch.clamp(img+noise, 0., 1.)
    return img


noise_transform = T.Lambda(add_noise)

def make_simclr_transforms(jitter_strength=0.5, blur=0., img_size=32):
    s = jitter_strength
    jitter = T.ColorJitter(
        brightness=0.8*s, contrast=0.8*s,
        saturation=0.8*s, hue=0.2*s)
    blurrer = T.GaussianBlur(kernel_size=img_size/10., sigma=(0.1, 1.0))

    multiple_transform = T.Compose([
        T.RandomResizedCrop(
            (img_size, img_size), scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(),
        T.RandomApply((jitter,), p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply((blurrer,), p=blur),
        T.Lambda(lambda x: x.contiguous()),
    ])

    return multiple_transform
