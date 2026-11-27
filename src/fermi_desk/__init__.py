"""Fermi Paradox Empirical Constraint & BOED desk."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _installed_version
from pathlib import Path

_DISTRIBUTION = "fermi-desk"


def _resolve_version() -> str:
    """The package version, from its single source of truth: ``[project].version``.

    In a source checkout (editable install) ``pyproject.toml`` is read directly, so the
    package, the CLI and the FastAPI app can never drift from it, even before the next
    ``uv sync`` refreshes the installed metadata. An installed wheel has no
    ``pyproject.toml`` beside it and reports its distribution metadata, which the build
    backend wrote from the same field.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if pyproject.is_file():
        try:
            project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            project = {}
        if project.get("name") == _DISTRIBUTION and isinstance(project.get("version"), str):
            return project["version"]
    try:
        return _installed_version(_DISTRIBUTION)
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()
