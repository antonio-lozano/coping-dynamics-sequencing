# Data layout (not tracked in git)

Set `COPING_DYNAMICS_DATA` to point here if your data live outside the repo.

```
data/
|-- metadata/
|   |-- index.csv
|   `-- config.yml
|-- aya_raw/
|   `-- *_filtered.h5
|-- freezing_predictions/
|   `-- *_freezing_predictions_only.csv
|-- processed/
|   |-- new_results.pkl
|   |-- new_results_clusters.pkl
|   |-- pose_features_with_clusters.parquet
`-- to_predict/
    `-- <dataset_id>/
        |-- *_matched_to_jen.csv
        |-- *DLC*.csv
        |-- fps_manifest.csv
        `-- mapping_report.json

results/
|-- model_training/
|   `-- models/
|       `-- xgb_behavior/
|           `-- <model_version>/
|               |-- model.json
|               |-- model.pkl
|               |-- metadata.json
|               |-- cv_metrics.json
|               |-- cv_confusion_matrix_oof.png
|               `-- cv_confusion_matrix_oof.pdf
`-- predictions/
    `-- <dataset_id>/
        `-- <model_version>/
            |-- predictions_frame.parquet
            |-- predictions_frame.csv
            |-- predictions_summary.csv
            `-- figures/
                |-- prediction_qc.png
                `-- prediction_qc.pdf
```

Use `scripts/preprocessing/map_aya_to_jen.py` to convert raw AYA tracking files into JEN-style CSVs under `data/to_predict/<dataset_id>/`.
Use `scripts/analysis/train_behavior_xgb.py` to train/save models, `scripts/analysis/predict_behavior_xgb.py` to run inference, and `scripts/analysis/visualize_behavior_predictions.py` to generate QC figures.
