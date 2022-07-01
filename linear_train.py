from model import LPLVGG11
import torch
from torch import nn
from lpl import LPLPass
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

device = torch.device("cuda:1")

model = LPLVGG11()
NAME = "lplvgg11_noPred"
model.load_state_dict(torch.load(f"models/{NAME}.pth"))
model.to(device)


cifar_ds = CIFAR10(root='../datasets/', transform=ToTensor(), train=True)
cifar_ds_test = CIFAR10(root='../datasets/', transform=ToTensor(), train=False)
dl = torch.utils.data.DataLoader(cifar_ds, batch_size=800)
dl_test = torch.utils.data.DataLoader(cifar_ds_test, batch_size=800)

with open(f"reports/{NAME}.txt", "w") as F:
    F.write(f"layer,accuracy\n")


TEST_LAYERS = [2, 5, 7, 10, 12, 15, 17, 20]
OUT_SIZES = [64*16*16, 128*8*8, 256*8*8, 256*4*4, 512*4*4, 512*2*2, 512*2*2, 512]

for i, layer in enumerate(TEST_LAYERS):
    submodel = model.model[:layer+1]
    print(submodel)

    linear = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(OUT_SIZES[i], 10).to(device)
    )
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        for images, labels in dl:
            images = images.to(device)
            labels = labels.to(device)

            representation = submodel(images)

            out = linear(representation)
            loss = criterion(out, labels)
            _, pred = torch.max(out, axis=1)
            acc = (pred == labels).float().mean().item()
            # print("Batch accuracy:", acc)
            # print("Loss", loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        accs = []
        for images, labels in dl_test:
            images = images.to(device)
            labels = labels.to(device)
            representation = submodel(images)

            out = linear(representation)
            _, pred = torch.max(out, axis=1)
            acc = (pred == labels).float().mean().item()
            accs.append(acc)

        print(f"Layer {layer}, epoch {epoch}")
        accuracy = sum(accs) / len(accs)
        print("Accuracy", accuracy)

    with open(f"reports/{NAME}.txt", "a") as F:
        F.write(f"{layer},{accuracy}\n")
