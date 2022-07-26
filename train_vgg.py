from lpl.model import LPLVGG11
import torch
import torchvision
from lpl.transform import make_simclr_transforms
from visdom import Visdom

EPOCHS_PER_LAYER = 70
LA_1 = 1.
LA_2 = 10.
PRED = 1.

device = torch.device("cuda:1")

model = LPLVGG11()
# model.load_state_dict(torch.load("models/STL_lplvgg11_noPred.pth"))
model.to(device)
n_layers = 8

ds = torchvision.datasets.STL10(
    root='../datasets/',
    transform=torchvision.transforms.ToTensor(),
    split='unlabeled'
)
dl = torch.utils.data.DataLoader(ds, batch_size=1600, num_workers=8, shuffle=True)

contrastive_transform = make_simclr_transforms(
    jitter_strength=0.5, blur=0.5, img_size=96)


for layer in range(n_layers):
    optimizer = torch.optim.Adam(
        model.weighted_layers[layer].parameters(),
        lr=1e-3, weight_decay=1.5e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    viz = Visdom()
    viz.line([0.], [0.], win='Hebbian', opts=dict(title='Hebbian loss'))
    viz.line([0.], [0.], win='Decorr', opts=dict(title='Decorr loss'))
    viz.line([0.], [0.], win='Predictive', opts=dict(title='Predictive loss'))
    viz.line([0.], [0.], win='LR', opts=dict(title='Learning rate'))

    step = 0
    for epoch in range(EPOCHS_PER_LAYER):
        print(f"OPT LAYER {layer}, EPOCH {epoch}")
        batch_count = 0
        for images, _ in dl:
            batch_count += 1
            step += 1
            images = images.to(device)

            out = model(contrastive_transform(images))  # first forward
            out = model(contrastive_transform(images))  # second forward

            lpl_layer = model.lpl_layers[layer]
            predictive = lpl_layer.predictive_loss() * PRED
            hebbian = lpl_layer.hebbian_loss() * LA_1
            decorr = lpl_layer.decorr_loss() * LA_2

            viz.line([hebbian.item()], [step], win='Hebbian', update='append')
            viz.line([decorr.item()], [step], win='Decorr', update='append')
            viz.line([predictive.item()], [step], win='Predictive', update='append')
            viz.line([scheduler.get_lr()], [step], win='LR', update='append')

            optimized_loss = hebbian + decorr + predictive
            optimized_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.reset()

        scheduler.step()
    torch.save(model.state_dict(), "models/STL_lplvgg11_70epochs.pth")
