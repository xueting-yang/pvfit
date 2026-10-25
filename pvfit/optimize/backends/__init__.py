"""Reconstruction programs the optimizer can drive; see :mod:`.base`."""

from .allen import AllenBackend
from .base import Backend, run_logged
from .pvfit import PVFitBackend

BACKENDS = {PVFitBackend.name: PVFitBackend, AllenBackend.name: AllenBackend}


def get_backend(name):
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"Unknown backend '{name}'; choose from {', '.join(BACKENDS)}"
        ) from None


__all__ = ["BACKENDS", "Backend", "get_backend", "run_logged"]
