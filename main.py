import torch
from torch import nn
from lpl import LPLPass
from data import noise_transform
from torchvision import transforms
from torchvision.datasets import MNIST

mnist_ds = MNIST('./data/', train=True, download=True,
                 transform=transforms.ToTensor())


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
    LPLPass(n_units=10, n_dims=1),
)

dl = torch.utils.data.DataLoader(mnist_ds, batch_size=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.cuda()

lambda1, lambda2 = 1., 10.

for images, labels in dl:
    images = images.cuda()

    out = model(noise_transform(images))  # first forward
    out = model(noise_transform(images))  # second forward

    hebbian = model[2].hebbian_loss()
    decorr = model[2].decorr_loss()
    predictive = model[2].predictive_loss()
    total_loss = lambda1 * hebbian + lambda2 * decorr + predictive
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(hebbian.item(), decorr.item(), predictive.item())
