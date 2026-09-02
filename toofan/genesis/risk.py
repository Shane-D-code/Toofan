"""TOOFAN risk-level semantics.

The Genesis module only defines a binary prediction:

    probability >= threshold -> genesis (class 1)
    probability <  threshold -> non-genesis (class 0)

To provide a ``risk_level`` for the standardized schema without inventing
arbitrary scientific semantics, we derive a *defensive* label purely from the
binary decision:  a storm flagged as genesis is labelled "GENESIS_LIKELY",
otherwise "NON_GENESIS".  These are NOT scientific severity grades; they are
logical descriptors derived directly from the documented binary decision rule.
"""

from __future__ import annotations

GENESIS = "GENESIS_LIKELY"
NON_GENESIS = "NON_GENESIS"


def risk_level_from_class(predicted_class: int) -> str:
    """Map a binary Genesis prediction to a defensive risk descriptor."""
    if int(predicted_class) == 1:
        return GENESIS
    return NON_GENESIS
