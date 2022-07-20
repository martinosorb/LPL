import torch
from torch import nn
from lpl import LPLPass
from transform import multiple_transform
from torchvision import transforms
from torchvision.datasets import CIFAR10
from visdom import Visdom

ds = CIFAR10('../datasets/', train=True, download=True,
             transform=transforms.ToTensor())


model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 500),
    # nn.ReLU(),
    LPLPass(),
)

dl = torch.utils.data.DataLoader(ds, batch_size=1024, num_workers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1.5e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=10, eta_min=1e-6)
model.cuda()

viz = Visdom()
viz.line([0.], [0.], win='Hebbian', opts=dict(title='Hebbian loss'))
viz.line([0.], [0.], win='Decorr', opts=dict(title='Decorr loss'))
viz.line([0.], [0.], win='Predictive', opts=dict(title='Predictive loss'))
viz.line([0.], [0.], win='LR', opts=dict(title='Learning rate'))

lambda1, lambda2 = 1., 10.
is_pred = False

step = 0
for epoch in range(50):
    print(f"== EPOCH {epoch}, LR {optimizer.param_groups[0]['lr']} ==")
    viz.line([optimizer.param_groups[0]['lr']], [epoch], win='LR', update='append')
    for images, labels in dl:
        images = images.cuda()

        out = model(multiple_transform(images))  # first forward
        out = model(multiple_transform(images))  # second forward

        hebbian = model[-1].hebbian_loss()
        decorr = model[-1].decorr_loss()
        predictive = model[-1].predictive_loss()
        total_loss = lambda1 * hebbian + lambda2 * decorr + is_pred * predictive

        viz.line([hebbian.item()], [step], win='Hebbian', update='append')
        viz.line([decorr.item()], [step], win='Decorr', update='append')
        viz.line([predictive.item()], [step], win='Predictive', update='append')
        step += 1

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(hebbian.item(), decorr.item(), predictive.item())

    scheduler.step()
    torch.save(model.state_dict(), "models/single_layer_noRelu_noPred.pth")
