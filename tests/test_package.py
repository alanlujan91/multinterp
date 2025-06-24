from __future__ import annotations

import importlib.metadata

import multinterp as m


def test_version() -> None:
    assert importlib.metadata.version("multinterp") == m.__version__
