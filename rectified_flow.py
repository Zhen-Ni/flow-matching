"""
Implements the Rectified Flow / Flow Matching logic.

This module provides the core algorithms for training and sampling
using Rectified Flow. It includes a Logit-Normal timestep sampler,
a forward process to interpolate between data and noise, and a
reverse process solver using Heun's Method (2nd order ODE).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def sample_timesteps(
    batch_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Sample timesteps using Logit-Normal distribution.

    Formula: u ~ N(mean, std^2), t = sigmoid(u)

    Args:
        batch_size: Number of timesteps to sample.
        device: Device to create the tensor on.

    Returns:
        Timesteps tensor of shape [batch_size].
    """
    # Sample u from normal distribution
    u = torch.randn(
        batch_size,
        device=device,
        dtype=torch.float32
    )

    # Map to (0, 1) using sigmoid
    t = torch.sigmoid(u)
    return t


def add_noise(
    x_1: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward process: Add noise to clean images (x_1) to get x_t.

    Formula:
        x_t = t * x_1 + (1 - t) * noise
        v_target = x_1 - noise

    Args:
        x_1: Clean images [B, C, H, W].
        t: Timesteps [B].

    Returns:
        x_t: Noisy images.
        v_target: Target velocity field.
    """
    noise = torch.randn_like(x_1)

    # Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
    t_b = t.view(t.shape[0], 1, 1, 1)

    # Linear interpolation
    x_t = t_b * x_1 + (1.0 - t_b) * noise

    # Target velocity is simply the direction from noise to data
    v_target = x_1 - noise

    return x_t, v_target


def _get_model_velocity(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: Optional[float] = None
) -> torch.Tensor:
    """Helper to get model velocity, handling CFG if needed."""
    if cfg_scale is not None and cfg_scale > 1.0:
        # Check if model has the CFG method
        if hasattr(model, 'forward_with_cfg'):
            return model.forward_with_cfg(x, t, y, cfg_scale)
        else:
            raise ValueError(
                "Model does not have 'forward_with_cfg' method. "
                "Cannot use Classifier-Free Guidance."
            )
    else:
        return model(x, t, y)


@torch.no_grad()
def step(
    model: nn.Module,
    x_t: torch.Tensor,
    t_now: float,
    t_next: float,
    y: torch.Tensor,
    cfg_scale: Optional[float] = None
) -> torch.Tensor:
    """Reverse process step using Heun's Method (2nd order ODE solver).

    1. Euler Step:
       d = t_next - t_now
       v_pred = model(x_t, t_now)
       x_next_euler = x_t + v_pred * d

    2. Heun Correction:
       v_next_pred = model(x_next_euler, t_next)
       v_avg = (v_pred + v_next_pred) / 2
       x_next = x_t + v_avg * d

    Args:
        model: The FlowMatchingModel.
        x_t: Current state [B, C, H, W].
        t_now: Current time (scalar float).
        t_next: Next time (scalar float).
        y: Class labels [B].
        cfg_scale: Optional classifier-free guidance scale.

    Returns:
        x_next: Updated state [B, C, H, W].
    """
    batch_size = x_t.shape[0]
    device = x_t.device
    dtype = x_t.dtype

    # Create timestep tensors
    t_now_tensor = torch.full(
        (batch_size,), t_now, device=device, dtype=dtype
    )
    t_next_tensor = torch.full(
        (batch_size,), t_next, device=device, dtype=dtype
    )

    # Calculate step size
    d = t_next - t_now

    # --- 1. Euler Step ---
    v_now = _get_model_velocity(
        model, x_t, t_now_tensor, y, cfg_scale
    )
    x_next_euler = x_t + v_now * d

    # --- 2. Heun Correction ---
    # Only apply correction if not at the last step (t_next < 1.0)
    # At t=1.0, the velocity should theoretically be 0, and the model
    # prediction might be unstable or unnecessary.
    if t_next < 1.0:
        v_next = _get_model_velocity(
            model, x_next_euler, t_next_tensor, y, cfg_scale
        )
        v_avg = (v_now + v_next) / 2.0
        x_next = x_t + v_avg * d
    else:
        x_next = x_next_euler

    return x_next


@torch.no_grad()
def generate(
    model: nn.Module,
    y: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: Optional[float] = None,
    img_shape: Tuple[int, int, int] = (3, 28, 28)
) -> torch.Tensor:
    """Full generation loop.

    1. Initialize x_0 from N(0, 1).
    2. Create time schedule (0 -> 1).
    3. Iterate steps using step().

    Args:
        model: The FlowMatchingModel.
        y: Class labels [B].
        num_steps: Number of sampling steps.
        cfg_scale: Optional classifier-free guidance scale.
        img_shape: Shape of the image (C, H, W).

    Returns:
        Generated images [B, C, H, W].
    """
    batch_size = y.shape[0]
    device = next(model.parameters()).device

    # 1. Initialize noise x_0
    x = torch.randn(batch_size, *img_shape, device=device)

    # 2. Create time schedule
    # t goes from 0 to 1. We need num_steps intervals.
    # e.g., steps=2 -> [0, 0.5, 1.0]
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

    # 3. Iterative sampling
    for i in range(num_steps):
        t_now = timesteps[i].item()
        t_next = timesteps[i + 1].item()

        x = step(model, x, t_now, t_next, y, cfg_scale)

    return x


if __name__ == "__main__":
    # Simple test to verify shapes
    from flow_matching_model import FlowMatchingModel

    # Setup
    device = "cpu"
    model = FlowMatchingModel(
        img_channel=3,
        img_height=28,
        img_width=28,
        patch_size=4,
        hidden_size=128
    ).to(device)

    # Test Generation
    print("Testing generation...")
    gen_y = torch.randint(0, 10, (2,)).to(device)
    # 直接调用 generate 函数，不再需要实例化类
    generated = generate(
        model,
        gen_y,
        num_steps=10,
        cfg_scale=1.5,
        img_shape=(3, 28, 28)
    )
    print(f"Generated shape: {generated.shape}")
