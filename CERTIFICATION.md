# Certification status

Current validated status:

```text
structural_regime_admissible=True
raw_structural_exact=False
exact_projective_certified=True
ctnet_structural_superiority=CALIBRATION_REQUIRED
certification_tier=CALIBRATION_REQUIRED
```

The raw calibrated CTNet transition is not yet exact. The current structural metrics are:

```text
amp_l2=0.43822440505
prob_l1=0.456387639046
spread_gap=0.00284448266029
mass_contrast_std=1.28768014908
```

The strong structural gate is:

```text
amp_l2 <= 1e-6
prob_l1 <= 1e-6
raw_structural_exact=True
```

The exact projective layer is separately certified:

```text
exact_projective_certified=True
```

This does not replace the missing raw structural calibration.
