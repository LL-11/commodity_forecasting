"""External data ingestion adapters."""

from .eia import EIAClient, normalize_henry_hub, normalize_storage

__all__ = ["EIAClient", "normalize_henry_hub", "normalize_storage"]
