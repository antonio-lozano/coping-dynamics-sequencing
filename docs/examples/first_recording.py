"""Run one complete bundled recording without modifying publication artifacts.

From the checkout root:
    uv run python docs/examples/first_recording.py --output ../coping-first-recording
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

from coping_dynamics.behavior_classifier import load_bundle, load_moseq_table, predict_behaviors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists() or output == root or root in output.parents:
        parser.error("Choose a new output directory outside the source checkout")

    source = root / "data/raw/moseq_syllables_per_frame.csv.gz"
    model = root / "classifier/figure7_behavior_classifier.joblib"
    frames = load_moseq_table(source)
    recording = sorted(frames["name"].dropna().unique())[0]
    # Retain the full recording: truncation would change rolling/lagged features.
    sample = frames.loc[frames["name"] == recording].copy()
    predictions = predict_behaviors(sample, load_bundle(model))
    expected = sample[["name", "frame_index"]].sort_values(["name", "frame_index"])
    actual = predictions[["name", "frame_index"]]
    if not actual.reset_index(drop=True).equals(expected.reset_index(drop=True)):
        raise RuntimeError("Prediction identifiers do not match the complete recording")
    if predictions["predicted_behavior"].isna().any():
        raise RuntimeError("Missing predicted labels")

    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    counts = predictions["predicted_behavior"].value_counts().rename_axis("behavior")
    counts.to_csv(output / "label_counts.csv", header=["frames"])
    evidence = {
        "recording": str(recording),
        "frames": len(sample),
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope": "single-recording saved-model inference; not independent accuracy validation",
    }
    (output / "run.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    print(f"Outputs: {output}")
