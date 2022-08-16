import torch


class LPLPass(torch.nn.Module):
    """
    This layer should do three things:
    - detach its output so that no backprop is allowed
    - keep track of means and variances to compute losses
    - provide utilities that compute the local losses

    Arguments:
        n_dims: dimensions of the input, excluding batch size
    """
    mse = torch.nn.MSELoss(reduction='sum')

    def __init__(self, global_average_pooling=False):
        super().__init__()
        self.current_z = None
        self.previous_z = None
        self.GAP = global_average_pooling

    def forward(self, z):
        this_out = torch.mean(z, dim=(-1, -2)) if self.GAP else z
        if self.current_z is not None:
            self.previous_z = self.current_z.detach()
        self.current_z = this_out
        return z.detach()

    def reset(self):
        self.current_z = None
        self.previous_z = None

    def predictive_loss(self):
        return 0.5 * self.mse(self.current_z, self.previous_z)  # looks good

    def hebbian_loss(self):
        var = torch.var(self.current_z - self.current_z.mean(0).detach(), dim=0)
        EPS = 1e-6  # TODO problematic. this depends intensely on epsilon
        return -0.5 * torch.log(var + EPS).sum()

    def decorr_loss(self):
        z = self.current_z
        batch_size = z.shape[0]
        n_neurons = z.shape[1]
        beta = 1./batch_size/(n_neurons-1)/4.

        centered_z_sq = (z - z.mean(0).detach()) ** 2  # bug fixed: mean along axis
        varmatrix = torch.einsum("bi,bj->ij", centered_z_sq, centered_z_sq)
        varmatrix.diagonal().zero_()  # bug fixed: wrong use of diagonal
        return beta * varmatrix.sum()  # bug fixed: removed 0.5x


class TimeSeriesLPL(torch.nn.Module):
    mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, z):
        self.current_z = z
        return z.detach()

    def predictive_loss(self, delta=1):
        SGz = self.current_z.detach()
        z = self.current_z
        tottime = len(z) - delta
        return 0.5 * self.mse(SGz[:-delta], z[delta:]) / tottime  # TODO check normalization by time is correct

    def hebbian_loss(self):
        zmean = self.current_z.detach().mean((0, 2))  # mean over batch and time
        var = torch.var(self.current_z - zmean, dim=(0, 2))  # same (check) TODO
        EPS = 1e-6  # TODO problematic. this depends intensely on epsilon
        return -0.5 * torch.log(var + EPS).sum()

    def decorr_loss(self):
        z = self.current_z
        batch_size, n_neurons, tottime = z.shape
        beta = 1./batch_size/(n_neurons-1)/tottime

        centered_z_sq = (z - z.mean(0).detach()) ** 2
        varmatrix = torch.einsum("bit,bjt->ij", centered_z_sq, centered_z_sq)
        varmatrix.diagonal().zero_()
        return beta * varmatrix.sum()
