import torch
from torch import nn
from lpl import LPLPass
from data import multiple_transform
from torchvision import transforms
from torchvision.datasets import CIFAR10

ds = CIFAR10('../datasets/', train=True, download=True,
             transform=transforms.ToTensor())


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 500),
    nn.ReLU(),
    LPLPass(),
)

dl = torch.utils.data.DataLoader(ds, batch_size=800)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model.cuda()

lambda1, lambda2 = 0.0, 1.
is_pred = True

for epoch in range(10):
    for images, labels in dl:
        images = images.cuda()

        out = model(multiple_transform(images))  # first forward
        out = model(multiple_transform(images))  # second forward

        hebbian = model[-1].hebbian_loss()
        decorr = model[-1].decorr_loss()
        predictive = model[-1].predictive_loss()
        total_loss = lambda1 * hebbian + lambda2 * decorr + is_pred * predictive
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(hebbian.item(), decorr.item(), predictive.item())
    torch.save(model.state_dict(), "models/single_layer_withRelu.pth")
