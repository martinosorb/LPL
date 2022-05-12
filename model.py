import torch
from torch import nn
from lpl import LPLPass
from torchvision.models import vgg11

RELU_IDX = [1, 4, 7, 9, 12, 14, 17, 19]
lambda1, lambda2 = 1., 10.


class LPLVGG11(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = vgg11(pretrained=False).features
        self.final_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        for idx in RELU_IDX:
            n_channels_conv = self.model[idx-1].out_channels
            self.model[idx] = nn.Sequential(
                nn.ReLU(inplace=True),
                LPLPass(n_units=n_channels_conv, global_average_pooling=True))

    def forward(self, x):
        return self.final_avgpool(self.model(x))

    def compute_lpl_losses(self):
        loss = torch.zeros(3, len(RELU_IDX))
        for i, idx in enumerate(RELU_IDX):
            lpl_layer = self.model[idx][1]
            loss[0, i] = lpl_layer.predictive_loss()
            loss[1, i] = lambda1 * lpl_layer.hebbian_loss()
            loss[2, i] = lambda2 * lpl_layer.decorr_loss()
        return loss
