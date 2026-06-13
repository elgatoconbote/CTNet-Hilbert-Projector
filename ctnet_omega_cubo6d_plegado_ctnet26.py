#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone compatibility shim for CTNet-Hilbert-Projector.

This file preserves the historical import path used by the Cubo-only solver while
redirecting the implementation to the local package copy. It makes this repository
self-contained and removes the need to run inside CTNet-2.6-Omega-Cubo-6D.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ctnet_hilbert_projector.ctnet_omega_core import *  # noqa: F401,F403
