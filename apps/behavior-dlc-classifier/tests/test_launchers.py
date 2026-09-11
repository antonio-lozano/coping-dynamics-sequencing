"""Execute platform launchers with disposable Python environments, without a GUI."""

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.mark.parametrize("local_environment", [True, False], ids=["local-venv", "path-python"])
@pytest.mark.parametrize("exit_code", [0, 7], ids=["success", "startup-failure"])
def test_launcher_environment_and_exit(tmp_path, local_environment, exit_code):
    app = tmp_path / "Behavior Studio with spaces"
    app.mkdir()
    windows = sys.platform == "win32"
    launcher_name = "launch_freezing_gui.bat" if windows else "launch_freezing_gui.sh"
    launcher = app / launcher_name
    shutil.copyfile(Path(__file__).parents[1] / launcher_name, launcher)
    environment = app / ".venv" if local_environment else tmp_path / "Python with spaces"
    venv.EnvBuilder(with_pip=False).create(environment)
    executable_dir = environment / ("Scripts" if windows else "bin")
    package = app / "src" / "freezing_dlc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "gui.py").write_text(
        "import json, os, sys\n"
        "print('LAUNCH_RESULT=' + json.dumps({'cwd': os.getcwd(), 'prefix': sys.prefix}))\n"
        f"sys.exit({exit_code})\n"
    )
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(name, None)
    # Exclude host uv and Python. Only the selected disposable interpreter and
    # platform utilities may be discovered; no dependency downloads can occur.
    utilities = str(Path(env["SYSTEMROOT"]) / "System32") if windows else "/usr/bin:/bin"
    env["PATH"] = utilities if local_environment else str(executable_dir) + os.pathsep + utilities
    command = (
        [env["COMSPEC"], "/d", "/c", str(launcher)] if windows else ["/bin/bash", str(launcher)]
    )
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        input="\n",
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    records = [
        line.removeprefix("LAUNCH_RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("LAUNCH_RESULT=")
    ]
    assert len(records) == 1, result.stdout + result.stderr
    record = json.loads(records[0])
    assert Path(record["cwd"]).resolve() == app.resolve()
    assert Path(record["prefix"]).resolve() == environment.resolve()
    assert ("could not start" in result.stdout) == bool(exit_code)
    if exit_code:
        assert "GUIDE.md" in result.stdout
