#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone compatibility shim for CTNet-Hilbert-Projector.

This file preserves the historical trainer/observer import path used by the
Cubo-only solver while redirecting the required observer bridge functions to the
local package copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ctnet_hilbert_projector.observer_bridge import (  # noqa: F401
    Observador,
    all_perspective_up_loss,
    batch_to_state,
    multiscale_up_loss,
)
