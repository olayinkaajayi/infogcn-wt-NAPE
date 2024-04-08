import numpy as np
import torch
import torch.nn as nn

class Inverse_line(nn.Module):
    """docstring for Inverse_line."""

    def __init__(self, a=1., b=1.):
        super(Inverse_line, self).__init__()
        self.a = nn.Parameter(torch.tensor([4.3]))#(torch.randn(1))
        self.b = nn.Parameter(torch.tensor([1.2]))#(torch.randn(1))
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(1. - 1./(self.a + self.b*x) )

class Inverse_log_line(nn.Module):
    """docstring for Inverse_log_line."""

    def __init__(self, a=1., b=1.):
        super(Inverse_log_line, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(1 - 1/torch.log(self.a + self.b*x) )


class Exp_zero_one(nn.Module):
    """docstring for Exp_zero_one."""

    def __init__(self, a=1., b=1.):
        super(Exp_zero_one, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.2]))#(torch.randn(1))
        self.b = nn.Parameter(torch.tensor([1.4]))#(torch.randn(1))
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(1. - torch.exp(-self.a - self.b*x) )

class Sine_zero_one(nn.Module):
    """docstring for Sine_zero_one."""

    def __init__(self, a=1., b=1.):
        super(Sine_zero_one, self).__init__()
        self.a = nn.Parameter(torch.tensor([0.2]))#(torch.randn(1))
        self.b = nn.Parameter(torch.tensor([1.4]))#(torch.randn(1))
        self.relu = nn.ReLU()

    def forward(self, x):

        return self.relu(torch.sin(self.a + self.b*x) )


class Sig_Linear(nn.Module):
    """docstring for Sig_Linear."""

    def __init__(self, in_dim=8):
        super(Sig_Linear, self).__init__()
        self.lin = nn.Linear(in_dim, in_dim)

    def forward(self, x):

        return torch.sigmoid(self.lin(x))


class Sig_Linear2(nn.Module):
    """docstring for Sig_Linear2."""

    def __init__(self, a=-0.01, b=1.0):
        super(Sig_Linear2, self).__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))

    def forward(self, x):

        return torch.sigmoid(-self.a - self.b*x)


class Sig_Linear3(nn.Module):
    """docstring for Sig_Linear3."""

    def __init__(self, in_dim=8):
        super(Sig_Linear3, self).__init__()
        self.a = nn.Parameter(torch.randn(in_dim))
        self.b = nn.Parameter(torch.randn(in_dim))
        # self.relu = nn.ReLU()

    def forward(self, x):

        return torch.sigmoid(-self.a - self.b*x)
