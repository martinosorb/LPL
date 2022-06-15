from model import LPLVGG11
import torch
import torchvision
from data import multiple_transform

EPOCHS_PER_LAYER = 2

model = LPLVGG11()
n_layers = 8

cifar_ds = torchvision.datasets.CIFAR10(
    root='../datasets/', transform=torchvision.transforms.ToTensor())
dl = torch.utils.data.DataLoader(cifar_ds, batch_size=800)
model.cuda()

for layer in range(n_layers):
    optimizer = torch.optim.Adam(
        model.weighted_layers[layer].parameters(),
        lr=1e-3, weight_decay=1.5e-6)

    for epoch in range(EPOCHS_PER_LAYER):
        batch_count = 0
        loss_tracker = torch.zeros(3, n_layers)  # 3 losses, 8 VGG layers
        for images, _ in dl:
            batch_count += 1
            images = images.cuda()

            out = model(multiple_transform(images))  # first forward
            out = model(multiple_transform(images))  # second forward

            losses = model.compute_lpl_losses(lambda1=1., lambda2=10.)
            loss_tracker += losses
            optimized_loss = losses[:, layer].sum()
            print(optimized_loss.item())
            optimized_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"OPT LAYER {layer}")
        print(loss_tracker / batch_count)
    torch.save(model.state_dict(), "models/lplvgg11.pth")
