# Archived behavior-model coordinate contract

The root 33-feature experimental classifier and the companion 719-feature
behavior classifier are separate models. This correction concerns only
`apps/behavior-dlc-classifier/src/freezing_dlc/behavior.py`.

## Evidence and correction

The companion model and `classifier/legacy_shap/xgb_model.pkl` have identical
SHA256 `dcc88056b0c87ff746061291daf5e1e5629dee252fbf2b71dcefb55490918782`.
The SHAP archive contains 2,638 rows with its exact 719-column schema. Their
14-marker coordinates are centered, but retain variable orientation and pixel
scale. This is evidence about archived model inputs; it does not identify their
original training/test membership.

The inherited companion builder additionally rotated tail-to-nose onto +x and
rescaled body length. On those archived coordinate rows it changed orientation
concentration from 0.183 to 1.0, with a median wrapped angle error of 1.236 radians,
despite still producing all 719 correctly named columns. Schema equality alone
was insufficient.

A centered-only correction and five archive-based regression checks were
recovered from the already-local historical side-branch commit `c3e7572`.
Only the coordinate-transform change was adopted; no unrelated branch was
merged. Normal behavior inference now uses `training_center`. Rotation/scaling
remains available only under the explicit `experimental_egocentric_scaled` mode;
using that transform with the archived model is not validated transfer.

## Real-data check

On the external Animal 2.4 DLC recording, both paths executed with 11,250 frames
and 719 features. The corrected path returned finite probabilities summing to
one within 1.20e-7. It changed 3,505 frame labels relative to the mismatched path.
That is a measured consequence of the compatibility fix, **not an accuracy
improvement claim**: independent reference labels were not supplied for this check.
No model or archived result artifact was rewritten.

The five recovered checks cover schema, centered/unrotated/unscaled geometry,
PCA angle, lag/rolling identities and default preprocessing. The sampled SHAP
rows are not a contiguous recording; passing these checks cannot establish all
historical temporal features, full training parity, or independent accuracy.
The complete original training implementation, split identities and assignment
provenance remain required. The existing freezing override and tracking-QC
policies are not changed by this coordinate correction.
