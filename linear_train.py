from model import LPLVGG11
import torch
from torch import nn
from lpl import LPLPass
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


model = LPLVGG11()
# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(3072, 500),
#     # nn.ReLU(),
#     # LPLPass(),
# )
model.load_state_dict(torch.load("models/lplvgg11_noPred.pth"))

cifar_ds = CIFAR10(root='../datasets/', transform=ToTensor(), train=True)
cifar_ds_test = CIFAR10(root='../datasets/', transform=ToTensor(), train=False)
dl = torch.utils.data.DataLoader(cifar_ds, batch_size=800)
dl_test = torch.utils.data.DataLoader(cifar_ds_test, batch_size=800)


linear = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(16384, 10).cuda()
)
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
model.cuda()

submodel = model.model[:8]

criterion = torch.nn.CrossEntropyLoss()
for epoch in range(10):
    print(f"EPOCH {epoch}")

    for images, labels in dl:
        images = images.cuda()
        labels = labels.cuda()

        representation = submodel(images)

        out = linear(representation)
        loss = criterion(out, labels)
        _, pred = torch.max(out, axis=1)
        acc = (pred == labels).float().mean().item()
        print("Batch accuracy:", acc)
        print("Loss", loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accs = []
    for images, labels in dl_test:
        images = images.cuda()
        labels = labels.cuda()
        representation = submodel(images)

        out = linear(representation)
        _, pred = torch.max(out, axis=1)
        acc = (pred == labels).float().mean().item()
        accs.append(acc)
    print("Acc", sum(accs)/len(accs))


