"""Read-only real-data probe; compare fresh-process NumPy dispatch modes.

Writes JSON only to stdout. Never executes a figure generator or changes the
historical ordering, model, reference files, or scientific acceptance threshold.
"""

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from numpy.core import _multiarray_umath

from scripts.derive_tables import fig6_resilience_stats as fig6

if "--worker" not in sys.argv:
    optional = [
        feature
        for feature in _multiarray_umath.__cpu_dispatch__
        if _multiarray_umath.__cpu_features__.get(feature, False)
    ]
    results = {}
    for mode in ("default", "optional_simd_disabled"):
        child_env = os.environ.copy()
        if mode == "optional_simd_disabled":
            child_env["NPY_DISABLE_CPU_FEATURES"] = ",".join(optional)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)
        results[mode] = json.loads(result.stdout)
        results[mode]["stderr"] = result.stderr
    print(json.dumps(results, indent=2, allow_nan=False))
else:
    source_before = hashlib.sha256(fig6.CSV.read_bytes()).hexdigest()
    df = fig6.load()
    orders = {}
    for keys, group in df.groupby(["Animal", "Experiment"]):
        order = group.sort_values("Time_bin").index.to_numpy(dtype="<i8")
        orders[json.dumps([str(key) for key in keys])] = hashlib.sha256(order.tobytes()).hexdigest()
    bouts = fig6.bout_table(df)
    statistics = fig6.build(
        [
            (cluster, fig6.contrasts(bouts[bouts.Cluster == cluster], "MeanBoutDuration"))
            for cluster in fig6.CLUSTERS
        ],
        "cluster",
    )
    runtime = io.StringIO()
    with redirect_stdout(runtime):
        np.show_runtime()
    source_after = hashlib.sha256(fig6.CSV.read_bytes()).hexdigest()
    if source_before != source_after:
        raise RuntimeError("Diagnostic changed its source input")
    cpu_model = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = sorted(
            {
                line.strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith(("model name", "flags"))
            }
        )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "cpu": cpu_model,
                "cpu_features": _multiarray_umath.__cpu_features__,
                "disabled_features": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
                "runtime": runtime.getvalue(),
                "input_sha256": source_before,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "analysis_sha256": hashlib.sha256(Path(fig6.__file__).read_bytes()).hexdigest(),
                "time_dtype": str(df.Time_bin.dtype),
                "rows": len(df),
                "tied_rows": int(
                    df.duplicated(["Animal", "Experiment", "Time_bin"], keep=False).sum()
                ),
                "sorted_row_hashes": orders,
                "bout_table_sha256": hashlib.sha256(bouts.to_csv(index=False).encode()).hexdigest(),
                "sniff_full_precision": fig6.contrasts(
                    bouts[bouts.Cluster == "Sniffing"], "MeanBoutDuration"
                ),
                "bout_statistics": statistics.to_dict(orient="records"),
            },
            indent=2,
            allow_nan=False,
        )
    )
