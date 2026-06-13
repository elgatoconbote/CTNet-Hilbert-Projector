from ctnet_hilbert_projector.ctnet_omega_core import FoldLayout, FoldedCTNetOmegaCubo26
from ctnet_hilbert_projector.observer_bridge import Observador, all_perspective_up_loss, batch_to_state


def test_standalone_observer_bridge_imports():
    assert FoldLayout is not None
    assert FoldedCTNetOmegaCubo26 is not None
    assert Observador is not None
    assert batch_to_state is not None
    assert all_perspective_up_loss is not None
