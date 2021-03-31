from typing import Callable, List
import torch
from torch import Tensor


def differential_equation(x: Tensor, function: Callable) -> Tensor:
    """Differential equation.

    Differential equation which wants to solve.

    Args:
        x (Tensor): Input value.
        function (Callable): Mapping

    Returns:
        Tensor: Output of differential equation. Target is zero-value.
    """
    x = x.clone().detach().requires_grad_(True)
    y_0 = function(x)
    y_1 = torch.autograd.grad(y_0.sum(), x, create_graph=True)[0]
    y_2 = torch.autograd.grad(y_1.sum(), x, create_graph=True)[0]
    return y_2 + y_0
    return y_2 + y_1 + y_0


def expression(x: Tensor, const: List[float] = [1, 1]):
    """Expression of the solution of the differential equation.

    Args:
        x (Tensor): Input value.
        const (List[float], optional): Const value. Defaults to [1, 1].

    Returns:
        Tensor: Output of formula.
    """
    return const[0] * torch.sin(x) + const[1] * torch.cos(x)
    return torch.exp(-x / 2) * (
        const[0] * torch.sin((3 ** 0.5 / 2) * x)
        + const[1] * torch.cos((3 ** 0.5 / 2) * x)
    )


if __name__ == '__main__':
    x = torch.randn(10000)
    error = differential_equation(x, expression)
    print(error.mean())
    print(error.var())
    print(error.abs().max())
    assert error.mean().abs() < 1e-6
    assert error.var() < 1e-12
    assert error.abs().max() < 1e-4
