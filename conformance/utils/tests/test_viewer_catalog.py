from __future__ import annotations

import json
import io
import tarfile
import threading
import hashlib
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "conformance/utils/src"))
import viewer_catalog
import package_fixtures
import unified_history
import unified_archives

import pytest

from conformance.utils.tests.fixture_test_support import run_node_contract




def _package():
    package_fixtures.package_snapshot("new", "new America/Los_Angeles", {}, {}, dry_run=False, prune=False)
