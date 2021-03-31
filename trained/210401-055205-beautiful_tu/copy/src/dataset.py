from typing import Callable, Tuple

import torch
from torch.functional import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm as tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_size: int,
        min: float,
        max: float,
        expression: Callable,
        noise_scale: float = 1,
    ):
        """Dataset class.

        Args:
            data_size (int): Dataset size
            min (float): Min of domain.
            max (float): Max of domain.
            expression (Callable): Solution of differential equation.
            noise (bool, optional): Whether to add noise to the data. Defaults to True.
        """
        self.min = min
        self.max = max

        x = self.generate_x(data_size)
        if noise_scale > 0:
            y = expression(x) + torch.randn(x.shape) * noise_scale
        else:
            y = expression(x)

        self.data_size = data_size
        self.data = {'x': x, 'y': y}

    def generate_x(self, size: int) -> Tensor:
        """Generates a sampled array from a uniform distribution in the domain.

        Args:
            size (int): Size of array.

        Returns:
            Tensor: Array of sampled value.
        """
        max = self.max
        min = self.min
        return (max - min) * torch.rand([size, 1]) + min

    def __len__(self):
        return self.data_size

    def __getitem__(self, index: int) -> Tuple:
        x = self.data['x'][index]
        y = self.data['y'][index]
        return x, y
