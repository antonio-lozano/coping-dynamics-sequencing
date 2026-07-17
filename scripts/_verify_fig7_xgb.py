"""Attempt to reproduce Fig 7 XGBoost behavioral classification.
NOTE: only 5 pose features are bundled (centroid_x/y, heading, angular_velocity,
velocity_px_s) -- not the full keypoint/pairwise-distance set the manuscript used.
We engineer rolling-window + lag features from these and run 3-fold stratified CV.
"""
import warnings, sys
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

CLUSTER_MAP = {"Freezing":[0,28],"Sniffing":[18,20],"Grooming":[24],
    "Turn":[1,3,5,6,10,15,26,27],"Locomotion":[11,12,14,16,19,21,25],
    "Climbing":[111],"Jump":[23,29,30,34]}
s2c = {s:c for c,ss in CLUSTER_MAP.items() for s in ss}

df = pd.read_csv("data/source/moseq_syllables_per_frame.csv.gz")
df = df.sort_values(["name","frame_index"]).reset_index(drop=True)
df["label"] = df.syllable.map(s2c).fillna("Unassigned")

base = ["centroid_x","centroid_y","heading","angular_velocity","velocity_px_s"]
df["abs_ang"] = df.angular_velocity.abs()
g = df.groupby("name", sort=False)
feats = base + ["abs_ang"]
# rolling window stats (per recording) + lags
for w in (5,15,30):
    for c in ["velocity_px_s","angular_velocity","abs_ang"]:
        df[f"{c}_rmean{w}"] = g[c].transform(lambda s: s.rolling(w,min_periods=1).mean())
        df[f"{c}_rstd{w}"]  = g[c].transform(lambda s: s.rolling(w,min_periods=1).std())
        feats += [f"{c}_rmean{w}", f"{c}_rstd{w}"]
for lag in (1,5,15):
    for c in ["velocity_px_s","angular_velocity","heading"]:
        df[f"{c}_lag{lag}"] = g[c].shift(lag)
        feats += [f"{c}_lag{lag}"]
df = df.fillna(0.0)

X = df[feats].values
le = LabelEncoder(); y = le.fit_transform(df.label.values)
print(f"rows={len(df)}  features={len(feats)}  classes={list(le.classes_)}")
print("class balance:", {c:int((df.label==c).sum()) for c in le.classes_})

clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.2,
                        tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                        n_jobs=-1, eval_metric="mlogloss")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
print("\nrunning 3-fold stratified CV (XGBoost)...")
pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=1)
acc = accuracy_score(y, pred)
print(f"\nOVERALL accuracy = {acc*100:.1f}%   (manuscript 62.7%)")
print(f"chance (1/{len(le.classes_)}) = {100/len(le.classes_):.1f}%   gain = {acc/(1/len(le.classes_)):.2f}x")
cm = confusion_matrix(y, pred)
print("\nper-behavior accuracy (recall):")
ms = {"Freezing":86.1,"Climbing":80.4,"Jump":70.6,"Sniffing":67.9,"Locomotion":61.5,
      "Grooming":45.0,"Turn":46.7,"Unassigned":30.1}
for i,c in enumerate(le.classes_):
    rec = cm[i,i]/cm[i].sum()*100
    print(f"  {c:<12} {rec:5.1f}%   (manuscript {ms.get(c,'?')})")
