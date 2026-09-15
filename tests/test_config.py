"""Importing the package must not create output directories."""

import os
import subprocess
import sys


def test_config_import_has_no_filesystem_side_effects(tmp_path):
    root = tmp_path / "untouched"
    env = os.environ.copy()
    env["COPING_DYNAMICS_ROOT"] = str(root)
    subprocess.run([sys.executable, "-c", "import coping_dynamics.config"], env=env, check=True)
    assert not root.exists()
