from model import LPLVGG11
import torch
import torch.nn.functional as F
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
import numpy as np

device = torch.device("cuda:1")

model = LPLVGG11()
NAME = "STL_lplvgg11_noPred"
model.load_state_dict(torch.load(f"models/{NAME}.pth"))
model.to(device)


ds = STL10(root='../datasets/', transform=ToTensor(), split='train')
ds_test = STL10(root='../datasets/', transform=ToTensor(), split='test')
dl = torch.utils.data.DataLoader(ds, batch_size=800)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=800)


AVGPOOL = False
SIZE_MUL = 9  # 9 for STL, 1 for CIFAR (not tested!)
TEST_LAYERS = [2, 5, 7, 10, 12, 15, 17, 20]
HW_SIZES = np.array([16, 8, 8, 4, 4, 2, 2, 1])
C_SIZES = np.array([64, 128, 256, 256, 512, 512, 512, 512])
OUT_SIZES = C_SIZES if AVGPOOL else C_SIZES*HW_SIZES*HW_SIZES*SIZE_MUL
print(max(OUT_SIZES))

report_name = f"reports/{NAME}_avgpool{AVGPOOL}.txt"
with open(report_name, "w") as rep:
    rep.write(f"layer,accuracy\n")


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
            if AVGPOOL:
                representation = F.adaptive_avg_pool2d(representation, (1, 1))

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
            if AVGPOOL:
                representation = F.adaptive_avg_pool2d(representation, (1, 1))

            out = linear(representation)
            _, pred = torch.max(out, axis=1)
            acc = (pred == labels).float().mean().item()
            accs.append(acc)

        print(f"Layer {layer}, epoch {epoch}")
        accuracy = sum(accs) / len(accs)
        print("Accuracy", accuracy)

    with open(report_name, "a") as rep:
        rep.write(f"{layer},{accuracy}\n")
