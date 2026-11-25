"""Domain-specific exceptions for invariant violations.

These deliberately do NOT subclass ``ValueError`` so that pydantic propagates them
unwrapped from validators: a caller sees ``UnitValidationError`` itself, not a
generic ``ValidationError`` that merely mentions it.
"""

from __future__ import annotations


class FermiDeskError(Exception):
    """Base class for all fermi-desk invariant violations."""


class SchemaValidationError(FermiDeskError):
    """A data file violates its schema contract."""


class UnitValidationError(SchemaValidationError):
    """A unit string is not a valid astropy unit, or a quantity has the wrong dimension."""


class Tier3SchemaError(SchemaValidationError):
    """A Tier-3 hypothesis is missing its quarantine_reason or monoculture_audit."""


class ProvenanceError(FermiDeskError):
    """A dataset's declared provenance (tier, checksum, citation) is missing or wrong."""


class DoubleCountingError(FermiDeskError):
    """Two surveys sharing a pointing_group_id were passed into one joint inference run."""


class Tier3ScoringError(FermiDeskError):
    """A quarantined (Tier-3) hypothesis was passed into empirical likelihood scoring."""


class GeometryError(FermiDeskError):
    """The galactic geometry was not specified explicitly, or is unsupported."""


class ComputationalBudgetError(FermiDeskError):
    """A Monte Carlo request exceeds the configured point budget (would be silently truncated otherwise)."""
