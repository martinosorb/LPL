import torch
from torch import nn
from .lpl import LPLPass
from torchvision.models import vgg11


class LPLVGG11(nn.Module):
    RELU_IDX = [1, 4, 7, 9, 12, 14, 17, 19]
    TEST_LAYERS = [2, 5, 7, 10, 12, 15, 17, 20]
    # image sizes of test layers assuming input=32x32
    HW_SIZES = [16, 8, 8, 4, 4, 2, 2, 1]
    C_SIZES = [64, 128, 256, 256, 512, 512, 512, 512]

    def __init__(self):
        super().__init__()

        self.model = vgg11(pretrained=False).features
        self.final_avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        for idx in self.RELU_IDX:
            self.model[idx] = nn.Sequential(
                nn.ReLU(inplace=True),
                LPLPass(global_average_pooling=True))

        self.lpl_layers = [self.model[idx][1] for idx in self.RELU_IDX]
        self.weighted_layers = [self.model[idx-1] for idx in self.RELU_IDX]

    def forward(self, x):
        return self.final_avgpool(self.model(x))

    def compute_lpl_losses(self, lambda1, lambda2, lambda_pred=1.):
        loss = torch.zeros(3, len(self.RELU_IDX))
        for i in range(len(self.RELU_IDX)):
            lpl_layer = self.lpl_layers[i]
            loss[0, i] = lambda_pred * lpl_layer.predictive_loss()
            loss[1, i] = lambda1 * lpl_layer.hebbian_loss()
            loss[2, i] = lambda2 * lpl_layer.decorr_loss()
        return loss

    def reset(self):
        for i in range(len(self.RELU_IDX)):
            self.lpl_layers[i].reset()
