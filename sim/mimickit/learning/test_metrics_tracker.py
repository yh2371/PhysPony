import torch


def _as_tensor(x, device=None):
    if torch.is_tensor(x):
        return x
    t = torch.tensor(x)
    if device is not None:
        t = t.to(device)
    return t


def compute_mpjpe(pred_pos: torch.Tensor, ref_pos: torch.Tensor) -> torch.Tensor:
    """
    Mean Per-Joint Position Error.

    Accepts tensors shaped:
      - (E, J, 3)
      - (T, E, J, 3)
    Returns:
      - scalar tensor (mean over all dims except xyz)
    """
    pred_pos = _as_tensor(pred_pos)
    ref_pos = _as_tensor(ref_pos, device=pred_pos.device)
    diff = pred_pos - ref_pos
    err = torch.linalg.vector_norm(diff, dim=-1)  # (..., J)
    return err.mean()


def compute_gmpjpe(pred_pos: torch.Tensor, ref_pos: torch.Tensor, root_idx: int = 0) -> torch.Tensor:
    """
    "Global" MPJPE variant where we remove root translation per-frame before computing MPJPE.
    This is a common sanity metric when global drift exists.
    """
    pred_pos = _as_tensor(pred_pos)
    ref_pos = _as_tensor(ref_pos, device=pred_pos.device)

    pred_centered = pred_pos - pred_pos[..., root_idx:root_idx + 1, :]
    ref_centered = ref_pos - ref_pos[..., root_idx:root_idx + 1, :]
    return compute_mpjpe(pred_centered, ref_centered)


def compute_velocity_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Generic L2 velocity loss. Works for a variety of shapes; it simply computes
    mean L2 difference over the last dimension.

    If pred/ref are sequences with time as dim 0, pass them directly.
    """
    pred = _as_tensor(pred)
    ref = _as_tensor(ref, device=pred.device)
    diff = pred - ref
    # If last dim is singleton (e.g. [..., D, 1]), squeeze it.
    if diff.shape[-1] == 1:
        diff = diff.squeeze(-1)
    return torch.mean(torch.square(diff))


def compute_acceleration_loss(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Generic L2 acceleration loss. If tensors include a time dimension, compute finite
    differences along time; otherwise falls back to velocity-style loss.
    """
    pred = _as_tensor(pred)
    ref = _as_tensor(ref, device=pred.device)

    # If we have an explicit time dimension, use second finite differences
    if pred.dim() >= 2 and pred.shape[0] >= 3 and pred.shape == ref.shape:
        pred_a = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
        ref_a = ref[2:] - 2.0 * ref[1:-1] + ref[:-2]
        return torch.mean(torch.square(pred_a - ref_a))

    return compute_velocity_loss(pred, ref)


class TestMetricsTracker:
    """
    Accumulates per-episode metrics during testing and returns global means.
    """

    def __init__(self, device):
        self._device = device
        self.reset()

    def reset(self):
        self._sum_mpjpe = torch.tensor(0.0, device=self._device)
        self._sum_gmpjpe = torch.tensor(0.0, device=self._device)
        self._sum_accel = torch.tensor(0.0, device=self._device)
        self._sum_vel = torch.tensor(0.0, device=self._device)
        self._sum_success = torch.tensor(0.0, device=self._device)
        self._sum_ep_len = torch.tensor(0.0, device=self._device)
        self._count = torch.tensor(0.0, device=self._device)
        return

    def update(self, mpjpe, gmpjpe, accel_loss, vel_loss, success_flag, ep_len):
        self._sum_mpjpe += _as_tensor(mpjpe, device=self._device)
        self._sum_gmpjpe += _as_tensor(gmpjpe, device=self._device)
        self._sum_accel += _as_tensor(accel_loss, device=self._device)
        self._sum_vel += _as_tensor(vel_loss, device=self._device)
        self._sum_success += _as_tensor(success_flag, device=self._device)
        self._sum_ep_len += _as_tensor(ep_len, device=self._device)
        self._count += 1.0
        return

    def get_metrics(self):
        denom = torch.clamp(self._count, min=1.0)
        return {
            "mpjpe": (self._sum_mpjpe / denom).item(),
            "gmpjpe": (self._sum_gmpjpe / denom).item(),
            "accel_loss": (self._sum_accel / denom).item(),
            "vel_loss": (self._sum_vel / denom).item(),
            "success_rate": (self._sum_success / denom).item(),
            "mean_ep_len": (self._sum_ep_len / denom).item(),
        }

