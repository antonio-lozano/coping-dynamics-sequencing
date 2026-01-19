#!/usr/bin/env python
"""
Run All Figures — Master script to generate all publication figures.

This script runs each figure generation script in sequence.
Use the flags below to control which figures to generate.

Usage:
    python scripts/run_all_figures.py           # Run all figures
    python scripts/run_all_figures.py --quick   # Skip slow figures (7, 8, 9)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# ==============================================================================
# CONFIGURATION — Set which figures to run
# ==============================================================================
RUN_FIGURE_3 = True   # Freezing analysis
RUN_FIGURE_4 = True   # Validation
RUN_FIGURE_5 = True   # Clusters
RUN_FIGURE_6 = True   # Diversity
RUN_FIGURE_7 = True   # Dynamics (transition matrices, BFL)
RUN_FIGURE_8 = True   # Resilience
RUN_FIGURE_9 = True   # SHAP explainability (slowest)

# ==============================================================================
# PATHS
# ==============================================================================
SCRIPTS_DIR = Path(__file__).resolve().parent / "generate_figures"

FIGURE_SCRIPTS = [
    ("Figure 3 - Freezing Analysis", "figure_3_freezing.py", RUN_FIGURE_3),
    ("Figure 4 - Validation", "figure_4_validation.py", RUN_FIGURE_4),
    ("Figure 5 - Behavioral Clusters", "figure_5_clusters.py", RUN_FIGURE_5),
    ("Figure 6 - Diversity Metrics", "figure_6_diversity.py", RUN_FIGURE_6),
    ("Figure 7 - Dynamics & BFL", "figure_7_dynamics.py", RUN_FIGURE_7),
    ("Figure 8 - Resilience Analysis", "figure_8_resilience.py", RUN_FIGURE_8),
    ("Figure 9 - SHAP Explainability", "figure_9_shap.py", RUN_FIGURE_9),
]

# ==============================================================================
# MAIN
# ==============================================================================
def run_script(name: str, script_path: Path, quick_mode: bool = False) -> tuple[bool, float]:
    """Run a single figure script and return success status and duration."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Script: {script_path.name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    try:
        # Set environment variables for figure scripts
        env = os.environ.copy()
        env["BATCH_MODE"] = "1"  # Skip plt.show()
        if quick_mode:
            env["QUICK_MODE"] = "1"  # Use minimal data for fast testing
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent.parent,  # repo root
            check=True,
            capture_output=False,  # Show output in real-time
            env=env,
        )
        duration = time.time() - start_time
        print(f"\n[OK] {name} completed in {duration:.1f}s")
        return True, duration
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n[FAIL] {name} FAILED after {duration:.1f}s (exit code: {e.returncode})")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n[ERROR] {name} ERROR: {e}")
        return False, duration


def main():
    # Parse command-line arguments
    quick_mode = "--quick" in sys.argv
    
    if quick_mode:
        print("\n[QUICK MODE] Skipping slow figures (7, 8, 9)")
        global RUN_FIGURE_7, RUN_FIGURE_8, RUN_FIGURE_9
        RUN_FIGURE_7 = False
        RUN_FIGURE_8 = False
        RUN_FIGURE_9 = False
    
    print("\n" + "="*70)
    print("  GENERATING ALL PUBLICATION FIGURES")
    print("="*70)
    
    # Update enabled flags after quick mode check
    scripts_to_run = [
        ("Figure 3 - Freezing Analysis", "figure_3_freezing.py", RUN_FIGURE_3),
        ("Figure 4 - Validation", "figure_4_validation.py", RUN_FIGURE_4),
        ("Figure 5 - Behavioral Clusters", "figure_5_clusters.py", RUN_FIGURE_5),
        ("Figure 6 - Diversity Metrics", "figure_6_diversity.py", RUN_FIGURE_6),
        ("Figure 7 - Dynamics & BFL", "figure_7_dynamics.py", RUN_FIGURE_7),
        ("Figure 8 - Resilience Analysis", "figure_8_resilience.py", RUN_FIGURE_8),
        ("Figure 9 - SHAP Explainability", "figure_9_shap.py", RUN_FIGURE_9),
    ]
    
    results = []
    total_start = time.time()
    
    for name, script_name, enabled in scripts_to_run:
        if not enabled:
            print(f"\n[SKIP] Skipping {name}")
            results.append((name, "SKIPPED", 0))
            continue
        
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            print(f"\n[WARN] Script not found: {script_path}")
            results.append((name, "NOT FOUND", 0))
            continue
        
        success, duration = run_script(name, script_path, quick_mode)
        results.append((name, "SUCCESS" if success else "FAILED", duration))
    
    total_duration = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n{'Figure':<40} {'Status':<12} {'Time':>10}")
    print("-"*62)
    
    n_success = 0
    n_failed = 0
    n_skipped = 0
    
    for name, status, duration in results:
        if status == "SUCCESS":
            status_str = "[OK] SUCCESS"
            n_success += 1
        elif status == "FAILED":
            status_str = "[X] FAILED"
            n_failed += 1
        elif status == "SKIPPED":
            status_str = "[-] SKIPPED"
            n_skipped += 1
        else:
            status_str = "[?] " + status
            n_skipped += 1
        
        time_str = f"{duration:.1f}s" if duration > 0 else "-"
        print(f"{name:<40} {status_str:<12} {time_str:>10}")
    
    print("-"*62)
    print(f"Total: {n_success} succeeded, {n_failed} failed, {n_skipped} skipped")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print("="*70 + "\n")
    
    # Exit with error code if any failed
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
