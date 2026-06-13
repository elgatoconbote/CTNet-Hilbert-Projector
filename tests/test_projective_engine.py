import torch

from ctnet_hilbert_projector.projective_engine import CTNetProjectiveState, sigma_to_index, index_to_sigma


def test_sigma_index_roundtrip():
    sigma = "upuppu"
    idx = sigma_to_index(sigma)
    assert index_to_sigma(idx, len(sigma)) == sigma


def test_projective_state_reads_amplitude(tmp_path):
    path = tmp_path / "state.pt"
    amplitudes = torch.zeros(64, dtype=torch.complex128)
    amplitudes[sigma_to_index("uuuuuu")] = 1.0 + 0.0j

    torch.save(
        {
            "amplitudes": amplitudes,
            "n": 6,
            "quantum_strong_certified": True,
            "final_omega_6d": 0.0,
            "final_omega_q": 1.9e-7,
            "normalization_error": 0.0,
            "phi_cubo_rad": 0.0,
        },
        path,
    )

    st = CTNetProjectiveState.load(path)
    assert st.is_certified() is True
    assert st.amplitude("uuuuuu") == 1.0 + 0.0j
    assert st.probability("uuuuuu") == 1.0
    assert st.normalization_error() == 0.0
