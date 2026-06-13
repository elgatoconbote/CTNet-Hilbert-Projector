from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def sigma_to_index(sigma: str) -> int:
    s = sigma.strip().replace(" ", "")
    if not s:
        raise ValueError("sigma vacia")
    idx = 0
    for k, ch in enumerate(s):
        if ch == "u":
            idx |= 1 << k
        elif ch == "p":
            pass
        else:
            raise ValueError(f"sigma invalida: {sigma!r}; usa solo u/p")
    return idx


def index_to_sigma(index: int, n: int) -> str:
    chars = []
    for k in range(n):
        chars.append("u" if ((index >> k) & 1) else "p")
    return "".join(chars)


@dataclass(frozen=True)
class ProjectiveReadout:
    sigma: str
    index: int
    amplitude: complex
    probability: float
    phase: float


class CTNetProjectiveState:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.amplitudes = torch.as_tensor(payload["amplitudes"]).to(torch.complex128)

        if "amplitudes_with_phi" in payload:
            self.amplitudes_with_phi = torch.as_tensor(payload["amplitudes_with_phi"]).to(torch.complex128)
        else:
            phi = float(payload.get("phi_cubo_rad", 0.0))
            eiphi = torch.exp(1j * torch.tensor(phi, dtype=torch.complex128))
            self.amplitudes_with_phi = eiphi * self.amplitudes

        self.n = int(payload.get("n", round(math.log2(int(self.amplitudes.numel())))))

    @classmethod
    def load(cls, path: str | Path) -> "CTNetProjectiveState":
        payload = torch.load(str(path), map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("El .pt no contiene un diccionario de estado")
        if "amplitudes" not in payload:
            raise KeyError("El .pt no contiene 'amplitudes'")
        return cls(payload)

    def amplitude(self, sigma: str, *, with_phi: bool = False) -> complex:
        idx = sigma_to_index(sigma)
        bank = self.amplitudes_with_phi if with_phi else self.amplitudes
        if idx >= bank.numel():
            raise IndexError(f"sigma={sigma!r} produce idx={idx}, fuera de {bank.numel()} amplitudes")
        a = bank[idx].detach().cpu()
        return complex(float(a.real), float(a.imag))

    def probability(self, sigma: str, *, with_phi: bool = False) -> float:
        a = self.amplitude(sigma, with_phi=with_phi)
        return float((a.real * a.real) + (a.imag * a.imag))

    def phase(self, sigma: str, *, with_phi: bool = False) -> float:
        a = self.amplitude(sigma, with_phi=with_phi)
        return math.atan2(a.imag, a.real)

    def delta_phase(self, sigma: str, tau: str, *, with_phi: bool = False) -> float:
        d = self.phase(sigma, with_phi=with_phi) - self.phase(tau, with_phi=with_phi)
        return math.atan2(math.sin(d), math.cos(d))

    def read(self, sigma: str, *, with_phi: bool = False) -> ProjectiveReadout:
        idx = sigma_to_index(sigma)
        a = self.amplitude(sigma, with_phi=with_phi)
        return ProjectiveReadout(
            sigma=sigma,
            index=idx,
            amplitude=a,
            probability=(a.real * a.real) + (a.imag * a.imag),
            phase=math.atan2(a.imag, a.real),
        )

    def normalization_error(self, *, with_phi: bool = False) -> float:
        bank = self.amplitudes_with_phi if with_phi else self.amplitudes
        return float((bank.abs().pow(2).sum() - 1.0).abs().detach().cpu())

    def certificate(self) -> dict[str, Any]:
        keys = [
            "quantum_strong_certified",
            "final_omega_6d",
            "final_omega_q",
            "final_amplitude_error",
            "final_probability_error",
            "final_phase_error",
            "final_exhaustive_error",
            "final_closure_error",
            "normalization_error",
            "phi_cubo_rad",
            "phi_cubo_over_pi",
            "eiphi_real",
            "eiphi_imag",
        ]
        return {k: self.payload.get(k) for k in keys if k in self.payload}

    def is_certified(self) -> bool:
        return bool(self.payload.get("quantum_strong_certified", False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Consulta amplitudes proyectivas CTNet u/p desde un .pt")
    ap.add_argument("state", help="Ruta al .pt guardado por solve_ising_cubo6d_only.py")
    ap.add_argument("sigma", nargs="*", help="Ramas u/p a consultar, por ejemplo uuuuuu pppppp upupup")
    ap.add_argument("--with-phi", action="store_true", help="Lee amplitudes con fase Cubo global aplicada")
    ap.add_argument("--certificate", action="store_true", help="Imprime el certificado guardado")
    if hasattr(ap, 'parse_intermixed_args'):
        args = ap.parse_intermixed_args()
    else:
        args = ap.parse_args()

    st = CTNetProjectiveState.load(args.state)

    if args.certificate:
        for k, v in st.certificate().items():
            print(f"{k}={v}")

    sigmas = args.sigma or ["u" * st.n]
    for sigma in sigmas:
        r = st.read(sigma, with_phi=args.with_phi)
        print(
            f"sigma={r.sigma} "
            f"index={r.index} "
            f"A.real={r.amplitude.real:.17g} "
            f"A.imag={r.amplitude.imag:.17g} "
            f"P={r.probability:.17g} "
            f"Theta={r.phase:.17g}"
        )

    print(f"normalization_error={st.normalization_error(with_phi=args.with_phi):.17g}")


if __name__ == "__main__":
    main()
