"""Genesis-specific exceptions.

These make the difference between "honest failure" and silent fallback
explicit.  A requested Genesis model that is missing or produces a
mis-formed prediction MUST surface as one of these errors — never as a
fabricated default prediction.
"""


class GenesisError(Exception):
    """Base class for all Genesis failures."""


class GenesisArtifactMissing(GenesisError):
    """A required Genesis artifact could not be located on disk."""


class GenesisFeatureError(GenesisError):
    """Input features do not match the expected Genesis feature schema."""


class GenesisModelUnavailable(GenesisError):
    """A requested Genesis model is unavailable (artifact missing/invalid)."""


class GenesisInsufficientInput(GenesisError):
    """Required Genesis input features are unavailable / insufficient."""
