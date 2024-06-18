from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
based on https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
'''

class LoraInjectedLinear(nn.Module):
    def __init__(
            self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r,out_features,bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std = 1/r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        
        freeze = self.linear(x)

        x = self.lora_down(x)
        x = self.selector(x)
        x = self.lora_up(x)
        x = self.dropout(x)

        return freeze + x * self.scale

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        assert diag.shape== (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)

        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(self.lora_up.weight.device).to(self.lora_up.weight.dtype)

class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        
        super().__init__()

        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        freeze = self.conv(x)

        x = self.lora_down(x)
        x = self.selector(x)
        x = self.lora_up(x)
        x = self.dropout(x)

        return freeze + x * self.scale

    def set_selector_from_diag(self, diag: torch.Tensor):
        # diag is a 1D tensor of size (r,)
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)

        # same device + dtype as lora_up
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)