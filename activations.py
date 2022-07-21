import torch
from lpl.model import LPLVGG11
import sys
from torchvision.transforms import ToTensor
from torchvision.datasets import STL10
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device("cuda")

ds_test = STL10(root='../datasets/', transform=ToTensor(), split='test')
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=800)
for images, labels in dl_test:
    images = images.to(device)
    break

for name in sys.argv[1:]:
    model = LPLVGG11()
    model.load_state_dict(torch.load(name))
    exp_name = Path(name).stem
    model.to(device)

    # define hooks
    recorded_activations = {}

    def hook_factory(n):
        def hook(module, inp, out):
            recorded_activations[n] = torch.mean(out).item()
        return hook

    for i in model.TEST_LAYERS:
        model.model[i].register_forward_hook(hook_factory(i))

    # do a single forward pass
    model(images)

    activations = [r for r in recorded_activations.values()]
    plt.plot(activations, label=exp_name)

plt.legend()
plt.show()
plt.savefig("activations.png")
