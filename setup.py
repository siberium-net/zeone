#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ZEONE Setup Script (Legacy Compatibility)
=========================================

[DEPRECATED] This file exists only for backward compatibility with old pip versions.
All configuration is in pyproject.toml (PEP 621).

For modern installations, use:
    pip install .
    pip install -e .[dev]
    pip install -e .[all]

This shim allows:
    python setup.py bdist_wheel  (legacy)
    python setup.py install      (legacy)
"""

from setuptools import setup

# All configuration is in pyproject.toml
# This is just a shim for old tooling
setup()

