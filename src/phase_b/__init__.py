"""Phase B decision and evaluation primitives.

The public API in this package follows the competition's discrete daily
dispatch contract.  It deliberately does not model a continuous battery LP:
the legal action space is small enough to enumerate exactly.
"""

from .contracts import (
    ACTIVE_WINDOW_COUNT,
    BLOCK_STEPS,
    CHARGE_POWER,
    CHARGE_START_MAX,
    DISCHARGE_POWER,
    DISCHARGE_START_MAX,
    IDLE_ACTION_COUNT,
    POWER_MAGNITUDE,
    STEPS_PER_DAY,
    ContractViolation,
    assert_valid_day_power,
    assert_valid_dispatch,
    validate_day_power,
    validate_dispatch,
)
from .block_model import (
    BLOCKS_PER_DAY,
    forward_block_means,
    make_forward_block_target,
    optimize_from_block_values,
)
from .dispatch import (
    DispatchDecision,
    build_day_power,
    enumerate_legal_windows,
    optimize_day,
    optimize_days,
)
from .evaluation import (
    evaluate_price_predictions,
    oracle_day,
    score_day,
)

__all__ = [
    "ACTIVE_WINDOW_COUNT",
    "BLOCK_STEPS",
    "BLOCKS_PER_DAY",
    "CHARGE_POWER",
    "CHARGE_START_MAX",
    "DISCHARGE_POWER",
    "DISCHARGE_START_MAX",
    "IDLE_ACTION_COUNT",
    "POWER_MAGNITUDE",
    "STEPS_PER_DAY",
    "ContractViolation",
    "DispatchDecision",
    "assert_valid_day_power",
    "assert_valid_dispatch",
    "build_day_power",
    "enumerate_legal_windows",
    "evaluate_price_predictions",
    "forward_block_means",
    "make_forward_block_target",
    "optimize_day",
    "optimize_days",
    "optimize_from_block_values",
    "oracle_day",
    "score_day",
    "validate_day_power",
    "validate_dispatch",
]
