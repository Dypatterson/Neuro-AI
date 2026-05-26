"""Phase 5: Hierarchical/structural retrieval mechanisms.

Following PROJECT_PLAN.md Phase 5 ("Structure and Abstraction"), this
package houses mechanisms that operate above single-scale Hopfield
retrieval — including HAM-style coupled multi-scale settling and (later)
layer-2 attractor structures fed by Phase 4 discoveries.
"""

from .natural_source_protocol import (
    DEFAULT_FREQUENCY_CAPS,
    SPECIAL_ATOMS,
    cap_label,
    eligible_triples_for_seed,
    fixedpoint_free_shuffle,
    protocol_for_frequency_cap,
    protocol_payload,
    same_scene_opportunities,
    select_queries,
    select_recommended_protocol,
    source_with_protocol_plan,
    validate_cleanup_preflight,
)

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
    "DEFAULT_FREQUENCY_CAPS",
    "M1Config",
    "RoleBindingStats",
    "SPECIAL_ATOMS",
    "cap_label",
    "eligible_triples_for_seed",
    "fixedpoint_free_shuffle",
    "protocol_for_frequency_cap",
    "protocol_payload",
    "run_m1_stack",
    "run_s2_weighted_mhn_check",
    "same_scene_opportunities",
    "select_queries",
    "select_recommended_protocol",
    "source_with_protocol_plan",
    "validate_cleanup_preflight",
]:
    if name in globals():
        __all__.append(name)
