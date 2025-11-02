#!/usr/bin/env python3
"""Regenerate VS Code C/C++ include paths for the active Python environment.

This script queries the active Python interpreter and NumPy to find the
appropriate header include directories and writes `.vscode/c_cpp_properties.json`.

Usage:
  python3 .vscode/update_includes.py

Run this after switching virtual environments so clangd/VS Code can resolve
`Python.h` and `numpy/arrayobject.h` in the editor.
"""
import json
import sys
import sysconfig
from pathlib import Path


def get_paths():
    try:
        import numpy as np
        numpy_inc = np.get_include()
    except Exception:
        numpy_inc = None

    include = sysconfig.get_path("include") or sysconfig.get_config_var("INCLUDEPY")
    return numpy_inc, include


def make_config(numpy_inc, py_inc):
    config = {
        "configurations": [
            {
                "name": "Mac",
                "includePath": [
                    "${workspaceFolder}/**",
                ],
                "defines": [],
                "macFrameworkPath": [
                    "/System/Library/Frameworks",
                    "/Library/Frameworks",
                ],
                "compilerPath": "/usr/bin/clang",
                "cStandard": "c11",
                "cppStandard": "c++17",
                "intelliSenseMode": "macos-clang-x64",
            }
        ],
        "version": 4,
    }
    incs = config["configurations"][0]["includePath"]
    if numpy_inc:
        incs.append(numpy_inc)
    if py_inc:
        incs.append(py_inc)
    return config


def main():
    numpy_inc, py_inc = get_paths()
    cfg = make_config(numpy_inc, py_inc)
    out = Path(__file__).resolve().parents[0] / "c_cpp_properties.json"
    out.write_text(json.dumps(cfg, indent=2))
    print("Wrote:", out)
    print("numpy include:", numpy_inc)
    print("python include:", py_inc)


if __name__ == "__main__":
    main()
