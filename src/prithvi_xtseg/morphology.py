from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LogitAdditiveClosing2d(nn.Module):
    """Per-class smooth additive closing on logits.

    This is the project's closing-like differentiable morphology adaptation,
    not a claim that every algebraic property of classical closing holds.
    """

    def __init__(
        self,
        num_classes: int = 11,
        kernel_size: int = 3,
        beta: float = 10.0,
        padding_mode: str = "replicate",
        se_initialization: str = "zeros",
    ) -> None:
        super().__init__()
        if int(kernel_size) != 3:
            raise ValueError("LogitAdditiveClosing2d currently requires kernel_size=3")
        if float(beta) <= 0:
            raise ValueError("beta must be positive")
        if padding_mode != "replicate":
            raise ValueError("Only replicate padding is supported")
        if se_initialization != "zeros":
            raise ValueError("Only zero/flat SE initialization is supported")
        self.num_classes = int(num_classes)
        self.kernel_size = int(kernel_size)
        self.padding_mode = str(padding_mode)
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32))
        self.structuring_element = nn.Parameter(
            torch.zeros(self.num_classes, 1, self.kernel_size, self.kernel_size)
        )

    def _windows(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if inputs.ndim != 4 or inputs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected [B,{self.num_classes},H,W], received {tuple(inputs.shape)}"
            )
        batch, channels, height, width = inputs.shape
        # Explicit boundary concatenation is deterministic and equals replicate
        # padding for the locked 3x3 structuring element.
        padded_width = torch.cat((inputs[..., :1], inputs, inputs[..., -1:]), dim=3)
        padded = torch.cat(
            (padded_width[:, :, :1, :], padded_width, padded_width[:, :, -1:, :]),
            dim=2,
        )
        windows = F.unfold(padded, kernel_size=self.kernel_size)
        windows = windows.reshape(
            batch, channels, self.kernel_size * self.kernel_size, height * width
        )
        return windows, (batch, channels, height, width)

    def smooth_dilation(self, inputs: torch.Tensor) -> torch.Tensor:
        windows, shape = self._windows(inputs)
        se = self.structuring_element.reshape(
            1, self.num_classes, self.kernel_size * self.kernel_size, 1
        ).to(dtype=inputs.dtype, device=inputs.device)
        beta = self.beta.to(dtype=inputs.dtype, device=inputs.device)
        return (torch.logsumexp(beta * (windows + se), dim=2) / beta).reshape(shape)

    def smooth_erosion(self, inputs: torch.Tensor) -> torch.Tensor:
        windows, shape = self._windows(inputs)
        se = self.structuring_element.reshape(
            1, self.num_classes, self.kernel_size * self.kernel_size, 1
        ).to(dtype=inputs.dtype, device=inputs.device)
        beta = self.beta.to(dtype=inputs.dtype, device=inputs.device)
        return (-torch.logsumexp(-beta * (windows - se), dim=2) / beta).reshape(shape)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        closed = self.smooth_erosion(self.smooth_dilation(inputs))
        if not torch.isfinite(closed).all():
            raise FloatingPointError("Smooth logit closing produced non-finite logits")
        return closed
