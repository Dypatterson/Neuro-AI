"""Loader for the empirical theta'(beta) calibration table (C.1.4).

The calibration JSON is produced by ``experiments/calibrate_theta_prime.py``
following the E1 protocol in
``notes/emergent-codebook/consolidation-geometry-diagnostic.md:172``.

The loader returns a callable ``theta_prime_fn(beta) -> float`` that callers
(e.g. ``per_atom_regime_diagnostics``) can pass in to replace the
``theta' ~= 1/beta`` starting approximation with the empirically-calibrated
boundary.

Behaviour summary:
- Exact ``beta`` match: returns the calibrated theta'.
- ``beta`` strictly between two calibrated points: log-beta linear interpolation.
- ``beta`` outside the calibrated range: falls back to ``1/beta`` and emits a
  one-shot warning to stderr per call.
- No calibration file: ``load_theta_prime_calibration`` returns ``None`` so
  the caller can fall back to ``lambda b: 1.0 / b`` itself.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

_DEFAULT_RELATIVE_PATH = Path("notes/emergent-codebook/theta_prime_calibration.json")


def _resolve_default_path() -> Path:
    """Return the repo-root-anchored default calibration path.

    The module lives at ``src/energy_memory/phase3/theta_prime_calibration.py``,
    so the repo root is three parents up.
    """
    return Path(__file__).resolve().parents[3] / _DEFAULT_RELATIVE_PATH


def _build_lookup(
    calibration: dict,
) -> List[Tuple[float, float]]:
    """Return sorted-by-beta list of (beta, theta_prime) tuples."""
    pairs: List[Tuple[float, float]] = []
    for beta_str, entry in calibration.items():
        try:
            beta = float(beta_str)
        except (TypeError, ValueError):
            continue
        try:
            theta_prime = float(entry["theta_prime"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(beta) or not math.isfinite(theta_prime):
            continue
        if beta <= 0.0:
            continue
        pairs.append((beta, theta_prime))
    pairs.sort(key=lambda p: p[0])
    return pairs


def load_theta_prime_calibration(
    path: Optional[Union[str, Path]] = None,
) -> Optional[Callable[[float], float]]:
    """Load calibration table and return a ``theta_prime_fn(beta)`` callable.

    Args:
        path: Optional path to a calibration JSON. When ``None``, defaults
            to ``notes/emergent-codebook/theta_prime_calibration.json`` at
            the repo root.

    Returns:
        A callable ``beta -> theta_prime`` if the file exists and contains
        at least one usable (beta, theta_prime) pair. Returns ``None`` when
        no calibration is available, so the caller can fall back to
        ``1/beta`` themselves.
    """
    if path is None:
        target = _resolve_default_path()
    else:
        target = Path(path)
    if not target.is_file():
        return None
    try:
        with target.open("r") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    calibration = raw.get("calibration") if isinstance(raw, dict) else None
    if not isinstance(calibration, dict):
        return None
    pairs = _build_lookup(calibration)
    if not pairs:
        return None
    betas = [p[0] for p in pairs]
    thetas = [p[1] for p in pairs]
    min_beta, max_beta = betas[0], betas[-1]

    def theta_prime_fn(beta: float) -> float:
        beta_f = float(beta)
        if beta_f <= 0.0:
            return float("inf")
        # Exact match (within float tolerance) returns the stored value.
        for b, t in pairs:
            if math.isclose(beta_f, b, rel_tol=1e-9, abs_tol=1e-12):
                return t
        # Outside calibrated range: fall back to 1/beta with a one-shot
        # warning. This is a hot path (called per-atom in
        # per_atom_regime_diagnostics) so we deliberately don't memoize:
        # the spec asks for "once per call".
        if beta_f < min_beta or beta_f > max_beta:
            print(
                f"[theta_prime_calibration] WARNING: beta={beta_f} outside "
                f"calibrated range [{min_beta}, {max_beta}]; falling back "
                f"to 1/beta.",
                file=sys.stderr,
            )
            return 1.0 / beta_f
        # Log-beta linear interpolation between the two surrounding pairs.
        log_b = math.log(beta_f)
        for i in range(1, len(pairs)):
            b_hi, t_hi = pairs[i]
            b_lo, t_lo = pairs[i - 1]
            if b_lo <= beta_f <= b_hi:
                log_lo = math.log(b_lo)
                log_hi = math.log(b_hi)
                if log_hi == log_lo:
                    return t_lo
                frac = (log_b - log_lo) / (log_hi - log_lo)
                return t_lo + frac * (t_hi - t_lo)
        # Should be unreachable given the in-range check above.
        return 1.0 / beta_f

    return theta_prime_fn
