# Aya<->Jen Mapping Explorer (DearPyGui)

Desktop GUI for side-by-side Jen (left) and Aya (right) keypoint inspection.

## Current Scope

- Pose file loading (`.csv`, `.h5`) for Jen and Aya.
- Aya view toggle:
  - `Raw Aya`
  - `Mapped to Jen` (live mapping via existing `map_aya_to_jen.py` logic)
- Coordinate view toggle:
  - `Raw px`
  - `Normalized pose` (same centroid/body-length preprocessing style used in robust ML pipeline)
- Live mapping editor (`JEN <- Aya`).
- Frame/time controls for left and right with optional link mode.
- Auto-pairing of files by normalized filename similarity.
- Mapping status panel at current Aya frame.

## Intentionally Not Implemented Yet

- Feature loading and feature plotting are intentionally blank placeholders.
- This is by design while the feature computation pipeline is being revised.

## Run (Conda env)

```powershell
C:\Users\admin\miniconda3\Scripts\conda.exe run --no-capture-output -n keypoint_moseq python gui/mapping_explorer_dpg/app.py
```

## Dependencies

- `dearpygui`
- `numpy`
- `pandas`

`numpy` and `pandas` already come from project environment; `dearpygui` is added in `environment.yml`.
