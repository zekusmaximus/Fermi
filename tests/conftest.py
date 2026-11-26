"""Shared offline fixtures (local data/ only; no network)."""

from __future__ import annotations

import shutil
import socket
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from fermi_desk.registry import TargetTable, load_factor_graph, load_targets

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------ network isolation
# AGENTS.md Sec. 4: the suite must run 100% offline. Every outbound (non-loopback)
# connection or DNS lookup raises; loopback stays allowed because Starlette's TestClient
# and the Windows asyncio event loop use local socket pairs.
_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}


class NetworkBlockedError(RuntimeError):
    pass


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    return isinstance(host, str) and (host in _LOOPBACK or host.startswith("127."))


def _guard(orig):
    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address):
            raise NetworkBlockedError(f"outbound network access blocked in tests: {address!r}")
        return orig(self, address)

    return connect


_orig_getaddrinfo = socket.getaddrinfo


def _getaddrinfo(host, *args, **kwargs):
    if not _is_loopback(host if isinstance(host, str) else str(host)):
        raise NetworkBlockedError(f"DNS lookup blocked in tests: {host!r}")
    return _orig_getaddrinfo(host, *args, **kwargs)


socket.socket.connect = _guard(socket.socket.connect)
socket.socket.connect_ex = _guard(socket.socket.connect_ex)
socket.getaddrinfo = _getaddrinfo


@pytest.fixture(scope="session")
def graph():
    return load_factor_graph(ROOT)


@pytest.fixture(scope="session")
def bl_targets(graph):
    return load_targets(graph.surveys["bl_enriquez_2017_692"], graph.root)


@pytest.fixture
def small_targets() -> TargetTable:
    xyz = np.array([[10.0, 3.0, 2.0], [-20.0, 5.0, -8.0], [0.0, -30.0, 12.0]]) * u.pc
    return TargetTable.from_arrays(["t1", "t2", "t3"], xyz)


@pytest.fixture
def data_copy(tmp_path: Path) -> Path:
    """Writable copy of the project data tree for negative-path schema tests."""
    shutil.copytree(ROOT / "data", tmp_path / "data")
    return tmp_path


def rate(x: float) -> u.Quantity:
    return x / (u.kpc**2 * u.Gyr)


def frac(x: float) -> u.Quantity:
    return x * u.dimensionless_unscaled
