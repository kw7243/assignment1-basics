"""Optimizer implementations for CS336 Assignment 1."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    """Clip the global L2 norm of parameter gradients.

    Args:
        parameters: Iterable of model parameters whose gradients
            will be clipped.
        max_l2_norm: Maximum permitted global gradient norm.
        eps: for scale = M/(||g|| +eps)

    Returns:
        None.

    Requires:
        max_norm > 0.
        Gradients are already computed by backward().
        All non-None gradients are compatible with their parameters.

    Effects:
        Modifies gradients in place if their global L2 norm exceeds
        max_norm. Does not modify parameter values.
        Parameters with no gradient are ignored.
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return None

    # Compute global L2 norm
    grad_norm = torch.sqrt(
        sum((g.float() ** 2).sum() for g in grads)
    )

    if grad_norm <= max_l2_norm:
        return None

    scale = max_l2_norm / (grad_norm + eps)

    for g in grads:
        g.mul_(scale)

    return None

class SGD(torch.optim.Optimizer):
    """SGD with a per-parameter learning rate that decays as 1 / sqrt(t + 1)."""

    # Rep:
    #   param_groups: Inherited list of dictionaries. Each group contains
    #       "params" (references to tensors) and "lr" (base learning rate).
    #   state[p]: Inherited dictionary for parameter p. May contain "t",
    #       the number of updates applied to p.
    #
    # Rep invariant:
    #   Each group's learning rate is nonnegative.
    #   For initialized state[p], t is a nonnegative integer.
    #
    # Abstraction function:
    #   Each group assigns a base learning rate alpha to its parameters.
    #   For a parameter p with gradient g and update count t, step()
    #   applies p <- p - alpha / sqrt(t + 1) * g, then increments t.
    #   Parameters without gradients are unchanged.
    #
    # Safety from rep exposure:
    #   PyTorch exposes mutable parameter references, param_groups, and state.
    #   External code must preserve the invariants of these objects.

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
    ) -> None:
        """Construct the step-decayed SGD optimizer.

        Args:
            params: Parameters to optimize, or dictionaries that define
                parameter groups with optional learning-rate overrides.
            lr: Nonnegative default learning rate for parameter groups.

        Raises:
            ValueError: If any supplied learning rate is negative.
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["lr"] < 0:
                raise ValueError(f"Invalid group learning rate: {group['lr']}")

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> None:
        """Apply one step of SGD to every parameter with a gradient.

        Args:
            closure: Optional callable that recomputes and returns the loss.
                It runs with gradient tracking enabled.

        Returns:
            Loss returned by closure, or None if no closure is supplied.

        Requires:
            Each non-None parameter gradient has a compatible shape and device.
            The caller has computed gradients or supplied a suitable closure.

        Effects:
            Updates parameters in place and increments the per-parameter
            count only for parameters with gradients.
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                p.add_(p.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1

        return None


class AdamW(torch.optim.Optimizer):
    """AdamW with decoupled weight decay (implementation scaffold)."""

    # Rep:
    #   param_groups: Inherited list of parameter-group dictionaries. Each
    #       group has "params", "lr", "betas", "eps", and "weight_decay".
    #   state[p]: Inherited per-parameter dictionary. Once initialized,
    #       stores an update count and first/second moment tensors.
    #
    # Rep invariant (to enforce in the completed implementation):
    #   lr >= 0, 0 <= beta1 < 1, 0 <= beta2 < 1, eps > 0,
    #   and weight_decay >= 0 for every parameter group.
    #   Each initialized moment tensor has the same shape as its parameter.
    #   Each initialized update count is a nonnegative integer.
    #
    # Abstraction function:
    #   Each parameter group specifies an AdamW update rule. Per-parameter
    #   state represents the update count and exponential moving averages
    #   of the gradient and squared gradient, used for adaptive updates.
    #   Weight decay is applied separately from the gradient update.
    #
    # Safety from rep exposure:
    #   PyTorch exposes mutable parameter references, param_groups, and state.
    #   External code must preserve the invariants of these objects.

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Construct an AdamW optimizer.

        Args:
            params: Parameters to optimize, or dictionaries defining
                parameter groups with optional hyperparameter overrides.
            lr: Nonnegative default learning rate.
            betas: Decay factors for first and second gradient moments.
            eps: Positive constant for numerical stability.
            weight_decay: Nonnegative coefficient for decoupled weight decay.

        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Apply one AdamW update to every parameter with a gradient.

        Args:
            closure: Optional callable that recomputes and returns the loss.
                It runs with gradient tracking enabled.

        Returns:
            Loss returned by closure, or None if no closure is supplied.

        Requires:
            Each non-None gradient is compatible with its parameter.
            The caller has computed gradients or supplied a suitable closure.

        Effects:
            After implementation, updates parameters and their moment state
            in place. Parameters without gradients remain unchanged.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups: # [{"params: [W], "lr": 0.1}, {"params: [b], "lr: 0.01}]
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # {t: int, m: p.shape, v: p.shape}
                t = state.get("t", 0)

                # Adjusted lr
                lr_t = lr*(1 - beta2**(t+1))**0.5 / (1 - beta1**(t+1))

                # Weight decay
                p.add_(p, alpha=-lr*weight_decay)

                # Moment updates
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                m.mul_(beta1).add_(p.grad, alpha= (1 - beta1)) # in_place ops

                v = state["v"]
                v.mul_(beta2).add_(p.grad**2, alpha= (1 - beta2))

                # Weight-update
                p.add_(m/(v**0.5 + eps), alpha=-lr_t)

                state["t"] = t + 1


        return loss
