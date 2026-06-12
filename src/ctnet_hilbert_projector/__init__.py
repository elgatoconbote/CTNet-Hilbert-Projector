"""CTNet Hilbert Projector.

Prototipo de simulacion cuantica estructural por base cardinal u/p.
"""

from .ctnet_omega_core import (
    AXES_6D,
    PLANES_6D,
    CUBO_VECTOR_DIM,
    FoldLayout,
    FoldedOmegaCuboState,
    FoldedCTNetOmegaCubo26,
)
from .hilbert_projector import (
    BranchReadout,
    HilbertProjectorConfig,
    HilbertProjection,
    UPPauli,
    UPHilbertProjector,
    enumerate_up_branches,
)

__all__ = [
    "AXES_6D",
    "PLANES_6D",
    "CUBO_VECTOR_DIM",
    "FoldLayout",
    "FoldedOmegaCuboState",
    "FoldedCTNetOmegaCubo26",
    "BranchReadout",
    "HilbertProjectorConfig",
    "HilbertProjection",
    "UPPauli",
    "UPHilbertProjector",
    "enumerate_up_branches",
]
