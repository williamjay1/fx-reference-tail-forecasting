"""Avoid a stalled Windows WMI lookup during pandas platform detection.

On this workstation, ``platform.machine()`` can block indefinitely while
querying WMI. Pandas asks for that value during import. The process
architecture is available directly from the standard Windows environment.
"""

from __future__ import annotations

import os
import platform
import sys

if sys.platform == "win32":
    _machine = os.environ.get("PROCESSOR_ARCHITECTURE")
    if not _machine:
        _machine = "AMD64" if sys.maxsize > 2**32 else "x86"
    platform.machine = lambda: _machine
    try:
        _windows_version = sys.getwindowsversion()
        _release = str(_windows_version.major)
        _version = str(_windows_version)
    except Exception:
        _release = ""
        _version = ""
    _uname = platform.uname_result(
        "Windows", os.environ.get("COMPUTERNAME", ""), _release, _version, _machine
    )
    platform.uname = lambda: _uname
    platform.system = lambda: "Windows"
    platform.processor = lambda: _machine
