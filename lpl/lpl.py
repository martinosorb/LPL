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
        self.GAP = global_average_pooling

    def forward(self, z):
        if self.current_z is None:
            self.current_z = torch.zeros_like(z[..., 0, 0] if self.GAP else z)

        self.previous_z = self.current_z.detach()
        self.current_z = torch.mean(z, dim=(-1, -2)) if self.GAP else z
        return z.detach()

    def predictive_loss(self):
        return 0.5 * self.mse(self.current_z, self.previous_z)  # looks good

    def hebbian_loss(self):
        var = torch.var(self.current_z - self.current_z.mean(0).detach(), dim=0)
        EPS = 1e-6  # TODO problematic. this depends intensely on epsilon
        return -torch.log(var + EPS).sum()

    def decorr_loss(self):
        z = self.current_z
        batch_size = z.shape[0]
        n_neurons = z.shape[1]
        beta = 1./batch_size/(n_neurons-1)

        centered_z_sq = (z - z.mean(0).detach()) ** 2  # bug fixed: mean along axis
        varmatrix = torch.einsum("bi,bj->ij", centered_z_sq, centered_z_sq)
        varmatrix.diagonal().zero_()  # bug fixed: wrong use of diagonal
        return beta * varmatrix.sum()  # bug fixed: removed 0.5x
