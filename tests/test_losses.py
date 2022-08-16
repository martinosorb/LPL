import torch
import sys
sys.path.append("lpl")
from lpl import LPLPass


lpl = LPLPass()


def test_zero_on_same():
    inp1 = torch.rand((1, 10))
    out1 = lpl(inp1)
    out2 = lpl(inp1)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, inp1)
    assert lpl.predictive_loss().item() == 0.


def test_loss_scaling():
    inp1 = torch.ones((10, 10))
    inp2 = torch.zeros((10, 10))

    # with small batch
    lpl(inp1[:2])
    lpl(inp2[:2])
    # L1 = lpl.predictive_loss().item()
    L2 = lpl.hebbian_loss().item()
    L3 = lpl.decorr_loss().item()
    lpl.reset()

    # with larger batch
    lpl(inp1)
    lpl(inp2)
    # assert L1 == lpl.predictive_loss().item()
    assert L2 == lpl.hebbian_loss().item()
    assert L3 == lpl.decorr_loss().item()


def test_repeated():
    inp1 = torch.rand((1, 10, 3, 3))
    out1 = lpl(inp1)
    out2 = lpl(inp1)
    assert lpl.predictive_loss().item() == 0.

    inp2 = torch.rand((1, 10, 3, 3))
    assert not torch.equal(inp1, inp2)

    out3 = lpl(inp2)
    assert lpl.predictive_loss().item() > 0.
    out4 = lpl(inp2)
    assert lpl.predictive_loss().item() == 0.


def test_delta_pred():
    some_linear = torch.nn.Linear(10, 12)
    inp1 = torch.rand((3, 10))
    inp2 = torch.rand((3, 10))
    out1 = lpl(some_linear(inp1))
    out2 = lpl(some_linear(inp2))
    assert not torch.equal(out1, out2)

    delta_w = -(out2 - out1).T @ inp2

    assert some_linear.weight.grad is None
    lpl.predictive_loss().backward()
    assert torch.equal(some_linear.weight.grad, -delta_w)


def test_delta_hebb():
    batch_size = 30
    some_linear = torch.nn.Linear(10, 12)
    inp1 = torch.rand((batch_size, 10))
    inp2 = torch.rand((batch_size, 10))
    out1 = lpl(some_linear(inp1))
    out2 = lpl(some_linear(inp2))
    assert not torch.equal(out1, out2)

    zmean = out2.mean(0).detach()
    alpha = 1 / (batch_size - 1)
    sigmasq = alpha * ((out2 - zmean)**2).sum(0)
    delta_w = ((out2 - zmean) / sigmasq).T @ inp2 * alpha  # too many alphas?

    assert some_linear.weight.grad is None
    lpl.hebbian_loss().backward()
    assert torch.allclose(some_linear.weight.grad, -delta_w, rtol=1e-3)


def test_delta_decorr():
    batch_size = 20
    n_neurons = 15
    some_linear = torch.nn.Linear(10, n_neurons)
    inp1 = torch.rand((batch_size, 10))
    inp2 = torch.rand((batch_size, 10))
    out1 = lpl(some_linear(inp1))
    out2 = lpl(some_linear(inp2))
    assert not torch.equal(out1, out2)

    zmean = out2.mean(0).detach()
    beta = 1. / batch_size / (n_neurons - 1)
    centered_z = out2 - zmean
    sumsq = (centered_z**2).sum(1, keepdim=True)
    delta_w = - (beta * centered_z * (sumsq - centered_z**2)).T @ inp2

    assert some_linear.weight.grad is None
    lpl.decorr_loss().backward()
    assert torch.allclose(some_linear.weight.grad, -delta_w, rtol=1e-3)
