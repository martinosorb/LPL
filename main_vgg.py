from model import LPLVGG11
import torch
import torchvision
from data import multiple_transform
from visdom import Visdom

EPOCHS_PER_LAYER = 10
LA_1 = 1.
LA_2 = 10.
PRED = 1.

model = LPLVGG11()
n_layers = 8

cifar_ds = torchvision.datasets.CIFAR10(
    root='../datasets/', transform=torchvision.transforms.ToTensor())
dl = torch.utils.data.DataLoader(cifar_ds, batch_size=800, num_workers=4)
model.cuda()

for layer in range(n_layers):
    optimizer = torch.optim.Adam(
        model.weighted_layers[layer].parameters(),
        lr=3e-4, weight_decay=1.5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-6)

    viz = Visdom()
    viz.line([0.], [0.], win='Hebbian', opts=dict(title='Hebbian loss'))
    viz.line([0.], [0.], win='Decorr', opts=dict(title='Decorr loss'))
    viz.line([0.], [0.], win='Predictive', opts=dict(title='Predictive loss'))
    viz.line([0.], [0.], win='LR', opts=dict(title='Learning rate'))

    step = 0
    for epoch in range(EPOCHS_PER_LAYER):
        batch_count = 0
        loss_tracker = torch.zeros(3, n_layers)  # 3 losses, 8 VGG layers
        for images, _ in dl:
            batch_count += 1
            step += 1
            images = images.cuda()

            out = model(multiple_transform(images))  # first forward
            out = model(multiple_transform(images))  # second forward

            losses = model.compute_lpl_losses(
                lambda1=LA_1, lambda2=LA_2, lambda_pred=PRED)
            loss_tracker += losses

            predictive, hebbian, decorr = losses[:, layer]
            viz.line([hebbian.item()], [step], win='Hebbian', update='append')
            viz.line([decorr.item()], [step], win='Decorr', update='append')
            viz.line([predictive.item()], [step], win='Predictive', update='append')

            optimized_loss = losses[:, layer].sum()
            print(optimized_loss.item())
            optimized_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"OPT LAYER {layer}")
        print(loss_tracker / batch_count)
    torch.save(model.state_dict(), "models/lplvgg11.pth")
