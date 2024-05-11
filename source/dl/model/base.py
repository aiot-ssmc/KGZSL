from torch import nn, Tensor


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor):
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.squeeze(dim=self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.unsqueeze(dim=self.dim)


class Take(nn.Module):
    def __init__(self, dim: int = 1, position: int = 0):
        super().__init__()
        self.dim = dim
        self.position = position

    def forward(self, x: Tensor) -> Tensor:
        return x.unbind(dim=self.dim)[self.position]
