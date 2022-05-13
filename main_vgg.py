from model import LPLVGG11
import torch
import torchvision
from data import multiple_transform


model = LPLVGG11()

cifar_ds = torchvision.datasets.CIFAR10(
    root='../datasets/', transform=torchvision.transforms.ToTensor())
dl = torch.utils.data.DataLoader(cifar_ds, batch_size=512)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.cuda()

for epoch in range(10):
    batch_count = 0
    loss_tracker = torch.zeros(3, 8)  # 3 losses, 8 VGG layers
    for images, _ in dl:
        batch_count += 1
        images = images.cuda()

        out = model(multiple_transform(images))  # first forward
        out = model(multiple_transform(images))  # second forward

        losses = model.compute_lpl_losses()
        loss_tracker += losses
        total_loss = losses.sum()
        total_loss.backward()
        print(total_loss.item())
        optimizer.step()
        optimizer.zero_grad()

    print(f"EPOCH {epoch}")
    print(loss_tracker / batch_count)
    torch.save(model.state_dict(), "models/lplvgg11.pth")
