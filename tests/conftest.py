"""Shared offline fixtures (local data/ only; no network)."""

from __future__ import annotations

import shutil
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from fermi_desk.registry import TargetTable, load_factor_graph, load_targets

ROOT = Path(__file__).resolve().parents[1]


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
