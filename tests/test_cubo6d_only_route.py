from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED = (
    "evolve_exact",
    "matrix_exp",
    "exact_certification",
    "global_phase",
    "transverse_field_ising_matrix",
    "IsingConfig",
)

def test_no_external_exact_or_hamiltonian_route():
    for base in ("src", "examples"):
        for p in (ROOT / base).rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            text = p.read_text(errors="ignore")
            hits = [x for x in BANNED if x in text]
            assert not hits, f"{p.relative_to(ROOT)} contains banned route tokens: {hits}"
