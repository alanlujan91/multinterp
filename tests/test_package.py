from __future__ import annotations

import importlib.metadata

import multinterp as m


def test_version():
    """Verify that the package version matches the metadata version."""
    assert importlib.metadata.version("multinterp") == m.__version__
