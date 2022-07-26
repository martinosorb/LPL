import torch
import sys
sys.path.append("lpl")
from lpl import LPLPass


lpl = LPLPass()


def test_zero_on_same():
    inp1 = torch.rand((1, 10, 3, 3))
    out1 = lpl(inp1)
    out2 = lpl(inp1)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, inp1)
    assert lpl.predictive_loss().item() == 0.


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
