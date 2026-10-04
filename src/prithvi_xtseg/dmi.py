from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class DMIDiagnostics:
    n_samples: int
    classes_present: int
    sign: float
    loss: float
    min_singular_value: float
    max_singular_value: float
    condition_number: float
    rank: int
    finite_loss: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def joint_numerator_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, int]:
    """Return Y^T P and the number of valid pixels for one micro-batch."""
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected logits [B,C,H,W] and target [B,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("Logit/target shape mismatch")
    probabilities = torch.softmax(logits.float(), dim=1)
    classes = int(probabilities.shape[1])
    labels = target.reshape(-1)
    flat = probabilities.permute(0, 2, 3, 1).reshape(-1, classes)
    valid = labels != int(ignore_index)
    labels = labels[valid].long()
    flat = flat[valid].to(torch.float64)
    if labels.numel() == 0:
        return torch.zeros((classes, classes), dtype=torch.float64, device=logits.device), 0
    if int(labels.min()) < 0 or int(labels.max()) >= classes:
        raise ValueError("Target class index is outside [0,num_classes)")
    one_hot = F.one_hot(labels, num_classes=classes).to(torch.float64)
    return one_hot.T @ flat, int(labels.numel())


def diagnostics_from_joint(
    joint: torch.Tensor,
    n_samples: int,
    classes_present: int,
    rank_rtol: float = 1e-12,
) -> tuple[torch.Tensor, DMIDiagnostics]:
    if joint.ndim != 2 or joint.shape[0] != joint.shape[1]:
        raise ValueError("DMI joint matrix must be square")
    sign, logabsdet = torch.linalg.slogdet(joint)
    loss = -logabsdet
    with torch.no_grad():
        singular = torch.linalg.svdvals(joint.detach())
        largest = singular.max()
        smallest = singular.min()
        rank = int((singular > largest * float(rank_rtol)).sum().cpu())
        condition = largest / smallest if smallest > 0 else torch.tensor(float("inf"), device=joint.device)
        diag = DMIDiagnostics(
            n_samples=int(n_samples),
            classes_present=int(classes_present),
            sign=float(sign.detach().cpu()),
            loss=float(loss.detach().cpu()),
            min_singular_value=float(smallest.cpu()),
            max_singular_value=float(largest.cpu()),
            condition_number=float(condition.cpu()),
            rank=rank,
            finite_loss=bool(torch.isfinite(loss).detach().cpu()),
        )
    return loss, diag


def validate_joint_for_gradient(joint: torch.Tensor, diagnostics: DMIDiagnostics) -> None:
    classes = int(joint.shape[0])
    if not diagnostics.finite_loss or diagnostics.sign == 0.0:
        raise FloatingPointError("DMI joint matrix has a non-finite log-determinant")
    if diagnostics.rank != classes or not np.isfinite(diagnostics.condition_number):
        raise FloatingPointError(
            f"DMI joint matrix is numerically rank deficient: rank={diagnostics.rank}/{classes}, "
            f"condition={diagnostics.condition_number}"
        )


def exact_joint_upstream_gradient(joint: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Return d[-log|det(Q)|]/d[Y^T P], with Q=(Y^T P)/N."""
    if int(n_samples) <= 0:
        raise ValueError("DMI requires at least one valid sample")
    identity = torch.eye(joint.shape[0], dtype=joint.dtype, device=joint.device)
    # d[-log|det(Q)|]/dQ = -Q^{-T}; Q = numerator/N.
    return -torch.linalg.solve(joint.T, identity) / float(n_samples)


def numpy_joint_and_diagnostics(
    probabilities: np.ndarray,
    target: np.ndarray,
    rank_rtol: float = 1e-12,
) -> tuple[np.ndarray, DMIDiagnostics]:
    """DMI diagnostics for a unique-pixel probability mosaic on CPU."""
    if probabilities.ndim != 2 or target.ndim != 1:
        raise ValueError("Expected probabilities [N,C] and target [N]")
    tensor_p = torch.from_numpy(np.asarray(probabilities, dtype=np.float64))
    tensor_y = torch.from_numpy(np.asarray(target, dtype=np.int64))
    classes = int(tensor_p.shape[1])
    one_hot = F.one_hot(tensor_y, num_classes=classes).to(torch.float64)
    joint = (one_hot.T @ tensor_p) / max(int(tensor_y.numel()), 1)
    _, diagnostics = diagnostics_from_joint(
        joint,
        n_samples=int(tensor_y.numel()),
        classes_present=int(torch.unique(tensor_y).numel()),
        rank_rtol=rank_rtol,
    )
    return joint.numpy(), diagnostics
