from __future__ import annotations

import importlib.metadata

import multinterp as m


def test_version():
    assert importlib.metadata.version("multinterp") == m.__version__
