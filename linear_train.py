from lpl.model import LPLVGG11
import torch
import torch.nn.functional as F
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Train linear decoders from all VGG layers.')
parser.add_argument('--name', type=str, help='The model to be tested')
parser.add_argument('--device', type=str, default='cuda', help='Device (cuda, cpu)')
parser.add_argument('--avgpool', action='store_true',
    help='Apply global average pooling to the layer before decoding.')
args = parser.parse_args()


model = LPLVGG11()
model.load_state_dict(torch.load(args.name))
exp_name = Path(args.name).stem
device = torch.device(args.device)
model.to(device)

ds = STL10(root='../datasets/', transform=ToTensor(), split='train')
ds_test = STL10(root='../datasets/', transform=ToTensor(), split='test')
dl = torch.utils.data.DataLoader(ds, batch_size=800)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=800)

AVGPOOL = args.avgpool
SIZE_MUL = 9  # 9 for STL, 1 for CIFAR (not tested!)
channel_n = np.asarray(model.C_SIZES)
maps_sizes_cifar = np.asarray(model.HW_SIZES)
OUT_SIZES = channel_n if AVGPOOL else channel_n*maps_sizes_cifar**2*SIZE_MUL
print(max(OUT_SIZES))

report_name = f"reports/{exp_name}_avgpool{AVGPOOL}.txt"
with open(report_name, "w") as rep:
    rep.write(f"layer,accuracy\n")


for i, layer in enumerate(model.TEST_LAYERS):
    submodel = model.model[:layer+1]
    print(submodel)

    linear = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(OUT_SIZES[i], 10).to(device)
    )
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(20):
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
