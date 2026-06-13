from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .ctnet_omega_core import FoldedCTNetOmegaCubo26, FoldedOmegaCuboState


@dataclass
class Observador:
    x: str
    y: str
    source: str
    regime: str = "zero_disk_online_text"


def _byte_signal(text: str, size: int, *, max_bytes: int = 2048) -> torch.Tensor:
    raw = (text or "").encode("utf-8", errors="ignore")[:max_bytes]
    if not raw:
        raw = b"<empty>"
    v = torch.zeros(size, dtype=torch.float32)
    for i, b in enumerate(raw):
        j = i % size
        depth = 1.0 + (i // size)
        v[j] += ((float(b) / 127.5) - 1.0) / math.sqrt(depth)
    phase = torch.linspace(0, 2.0 * math.pi, size, dtype=torch.float32)
    v = torch.tanh(v + 0.015 * torch.sin(phase) + 0.0075 * torch.cos(2.0 * phase))
    return v


def _text_tensor(text: str, shape: Tuple[int, ...], *, amp: float = 1.0, max_bytes: int = 2048) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    return (amp * _byte_signal(text, n, max_bytes=max_bytes)).reshape(*shape)


def _pad_anchor(batch: int, pad_size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if pad_size <= 0:
        return torch.zeros(batch, 0, dtype=dtype, device=device)
    phase = torch.linspace(0, 2.0 * math.pi, pad_size, dtype=dtype, device=device)
    pad = 0.01 * (torch.sin(phase) + 0.5 * torch.cos(2.0 * phase))
    return pad.unsqueeze(0).repeat(batch, 1)


def batch_to_state(
    model: FoldedCTNetOmegaCubo26,
    samples: List[Observador],
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_bytes: int,
) -> Tuple[FoldedOmegaCuboState, torch.Tensor, List[str]]:
    L = model.layout
    batch = len(samples)
    if batch < 1:
        raise ValueError("batch_to_state requires at least one Observador")

    z_rows = []
    mem_rows = []
    rel_rows = []
    target_z_rows = []
    regimes = []

    for ex in samples:
        z_rows.append(_text_tensor(f"<regime>{ex.regime}</regime>\n{ex.x}", (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        mem_rows.append(
            _text_tensor(
                f"<source>{ex.source}</source>\n<regime>{ex.regime}</regime>\n{ex.x}",
                (L.mem_slots, L.mem_dim),
                amp=0.01,
                max_bytes=max_bytes,
            )
        )
        rel_rows.append(
            _text_tensor(
                f"<relations>{ex.regime}|{ex.source}</relations>\n{ex.x[:1024]}",
                (L.rel_edges, L.rel_dim),
                amp=0.01,
                max_bytes=max_bytes,
            )
        )
        target_z_rows.append(_text_tensor(ex.y, (L.z_tokens, L.z_dim), amp=1.0, max_bytes=max_bytes))
        regimes.append(ex.regime)

    z = torch.stack(z_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    memory = torch.stack(mem_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    relations = torch.stack(rel_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    target_z = torch.stack(target_z_rows, dim=0).to(device=device, dtype=dtype, non_blocking=True)
    pad = _pad_anchor(batch, L.pad_size, dtype=dtype, device=device)

    with torch.no_grad():
        cubo0 = model.cubo(z, memory, relations)["vector"].to(device=device, dtype=dtype)

    return FoldedOmegaCuboState(z=z, memory=memory, relations=relations, cubo=cubo0, pad=pad), target_z, regimes


def _even_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2 == 0:
        return x
    return F.pad(x, (0, 1))


def _up_mse_last_dim(x: torch.Tensor) -> torch.Tensor:
    x = _even_last_dim(x)
    d2 = x.shape[-1] // 2
    u = x[..., :d2]
    p = x[..., d2:]
    return F.mse_loss(u, p)


def _pool_tokens(x: torch.Tensor, scale: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] < scale:
        return x
    b, n, d = x.shape
    usable = (n // scale) * scale
    if usable <= 0:
        return x
    return x[:, :usable, :].reshape(b, usable // scale, scale, d).mean(dim=2)


def multiscale_up_loss(x: torch.Tensor, *, token_scales: Tuple[int, ...] = (2, 4, 8)) -> torch.Tensor:
    terms: List[torch.Tensor] = [_up_mse_last_dim(x)]

    for shift in (1, 2, 3):
        if x.shape[-1] > shift:
            terms.append(_up_mse_last_dim(torch.roll(x, shifts=shift, dims=-1)))

    if x.ndim == 3:
        for shift in (1, 2, 4):
            if x.shape[1] > shift:
                terms.append(_up_mse_last_dim(torch.roll(x, shifts=shift, dims=1)))
        for scale in token_scales:
            if x.shape[1] >= scale:
                pooled = _pool_tokens(x, scale)
                terms.append(_up_mse_last_dim(pooled))
                if pooled.shape[1] > 1:
                    terms.append(_up_mse_last_dim(torch.roll(pooled, shifts=1, dims=1)))

    return torch.stack(terms).mean()


def all_perspective_up_loss(
    model: FoldedCTNetOmegaCubo26,
    state: FoldedOmegaCuboState,
    out: FoldedOmegaCuboState,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    xi_in = model.pack(state)
    xi_out = model.pack(out)
    delta = xi_out - xi_in

    z_up = multiscale_up_loss(out.z)
    mem_up = multiscale_up_loss(out.memory)
    rel_up = multiscale_up_loss(out.relations)
    cubo_up = multiscale_up_loss(out.cubo)
    xi_up = multiscale_up_loss(xi_out)
    delta_up = multiscale_up_loss(delta)

    total = torch.stack([z_up, mem_up, rel_up, cubo_up, xi_up, delta_up]).mean()
    metrics = {
        "up_total": float(total.detach().cpu()),
        "up_z": float(z_up.detach().cpu()),
        "up_memory": float(mem_up.detach().cpu()),
        "up_relations": float(rel_up.detach().cpu()),
        "up_cubo": float(cubo_up.detach().cpu()),
        "up_xi": float(xi_up.detach().cpu()),
        "up_delta": float(delta_up.detach().cpu()),
    }
    return total, metrics
