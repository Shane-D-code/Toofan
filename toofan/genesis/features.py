"""Genesis feature schema.

The Genesis models expect a fixed 34-feature representation.  This schema
declares the canonical ordering used at training/inference time.  The exact
feature list observed in the original Genesis training code follows the
documented 34-feature input.  The feature ORDER here MUST match the order in
which the installed artifact expects its inputs.

Missing/unordered input features are rejected via ``GenesisFeatureError`` —
the adapter never silently fills or reorders features.
"""

from __future__ import annotations

from typing import List

GENESIS_FEATURE_COUNT = 34

# Canonical ordered feature schema (matching the trained Genesis models).
GENESIS_FEATURES: List[str] = [
    "candidate_latitude",
    "candidate_longitude",
    "month_sin",
    "month_cos",
    "sst",
    "sst_anomaly",
    "tchp",
    "ohc700",
    "wind850",
    "wind200",
    "vort850",
    "vort200",
    "div850",
    "div200",
    "shear_mag",
    "shear_u",
    "shear_v",
    "rh500",
    "rh700",
    "rh850",
    "speed_850",
    "speed_700",
    "speed_500",
    "omega500",
    "pwat",
    "cape",
    "mslp",
    "wind_surface",
    "sst_grad",
    "dist2land",
    "latitude_anomaly",
    "longitude_anomaly",
    "tempo_sst_anom",
    "pres_anomaly",
]


def validate_feature_schema(features: List[str]) -> None:
    """Raise ``GenesisFeatureError`` if ``features`` is not the canonical set."""
    if list(features) != GENESIS_FEATURES:
        from .errors import GenesisFeatureError

        raise GenesisFeatureError(
            "Genesis feature schema mismatch. Expected 34 ordered features "
            "matching GENESIS_FEATURES; got %d feature(s)."
            % len(features)
        )
