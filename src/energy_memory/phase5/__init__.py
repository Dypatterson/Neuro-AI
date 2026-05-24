"""Phase 5: Hierarchical/structural retrieval mechanisms.

Following PROJECT_PLAN.md Phase 5 ("Structure and Abstraction"), this
package houses mechanisms that operate above single-scale Hopfield
retrieval — including HAM-style coupled multi-scale settling and (later)
layer-2 attractor structures fed by Phase 4 discoveries.
"""

try:  # pragma: no cover - exercised only when torch is available
    from .m1_role_energy import (
        M1Config,
        RoleBindingStats,
        run_m1_stack,
        run_s2_weighted_mhn_check,
    )
except ModuleNotFoundError:  # pragma: no cover
    pass

__all__ = []
for name in [
    "M1Config",
    "RoleBindingStats",
    "run_m1_stack",
    "run_s2_weighted_mhn_check",
]:
    if name in globals():
        __all__.append(name)
