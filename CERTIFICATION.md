# Exact Projective Certification

This repository now separates two validation layers.

## 1. Structural calibrated regime

This is the current CTNet transition regime tested against exact transverse-field Ising.
It reports approximate structural metrics such as:

```text
amp_l2
prob_l1
spread_gap
mass_contrast_std
structural_regime_admissible
```

The previous thresholds `amp_l2 < 0.50` and `prob_l1 < 0.50` are only admissibility thresholds for this calibrated structural regime.

## 2. Exact projective certificate

The exact layer checks the formal projective realization:

```text
Pi_quant(T_U(Xi)) = U Pi_quant(Xi)
```

and the mass-phase reconstruction:

```text
mu(sigma)    = |A(sigma)|^2
P(sigma)     = mu(sigma) / sum_tau mu(tau)
Theta(sigma) = arg A(sigma)
A_rec(sigma) = sqrt(P(sigma)) exp(i Theta(sigma))
```

The certificate requires:

```text
exact_amp_l2                       <= 1e-6
exact_prob_l1                      <= 1e-6
exact_observable_abs               <= 1e-6
exact_normalization_error          <= 1e-6
exact_projective_commutation_error <= 1e-6
exact_projective_certified=True
```

The main benchmark now prints both layers:

```bash
"$PY" examples/benchmark_projective_superiority.py \
  --cuda \
  --preset best \
  --validate-n 6 \
  --steps 3 \
  --dt 0.05 \
  --J 1.0 \
  --h 0.5 \
  --sizes 6 8 10 12 16 20 24 30 40
```

Expected key output:

```text
structural_regime_admissible=True
exact_projective_certified=True
ctnet_structural_superiority=PASS
certification_tier=EXACT_PROJECTIVE
```
