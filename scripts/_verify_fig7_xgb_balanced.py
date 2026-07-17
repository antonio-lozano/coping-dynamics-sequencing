import warnings,sys; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
CM={"Freezing":[0,28],"Sniffing":[18,20],"Grooming":[24],"Turn":[1,3,5,6,10,15,26,27],
    "Locomotion":[11,12,14,16,19,21,25],"Climbing":[111],"Jump":[23,29,30,34]}
s2c={s:c for c,ss in CM.items() for s in ss}
df=pd.read_csv("data/source/moseq_syllables_per_frame.csv.gz").sort_values(["name","frame_index"]).reset_index(drop=True)
df["label"]=df.syllable.map(s2c).fillna("Unassigned"); df["abs_ang"]=df.angular_velocity.abs()
g=df.groupby("name",sort=False); feats=["centroid_x","centroid_y","heading","angular_velocity","velocity_px_s","abs_ang"]
for w in (5,15,30):
    for c in ["velocity_px_s","angular_velocity","abs_ang"]:
        df[f"{c}_rm{w}"]=g[c].transform(lambda s:s.rolling(w,min_periods=1).mean())
        df[f"{c}_rs{w}"]=g[c].transform(lambda s:s.rolling(w,min_periods=1).std()); feats+=[f"{c}_rm{w}",f"{c}_rs{w}"]
for lag in (1,5,15):
    for c in ["velocity_px_s","angular_velocity","heading"]:
        df[f"{c}_l{lag}"]=g[c].shift(lag); feats+=[f"{c}_l{lag}"]
df=df.fillna(0.0)
X=df[feats].values; le=LabelEncoder(); y=le.fit_transform(df.label.values)
cv=StratifiedKFold(3,shuffle=True,random_state=42); pred=np.zeros_like(y)
for tr,te in cv.split(X,y):
    sw=compute_sample_weight("balanced",y[tr])
    m=xgb.XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.2,tree_method="hist",
                        subsample=0.8,colsample_bytree=0.8,n_jobs=-1,eval_metric="mlogloss")
    m.fit(X[tr],y[tr],sample_weight=sw); pred[te]=m.predict(X[te])
cm=confusion_matrix(y,pred)
print(f"BALANCED-weight XGBoost: overall acc={accuracy_score(y,pred)*100:.1f}%  balanced acc={balanced_accuracy_score(y,pred)*100:.1f}%  (manuscript overall 62.7%)")
ms={"Freezing":86.1,"Climbing":80.4,"Jump":70.6,"Sniffing":67.9,"Locomotion":61.5,"Grooming":45.0,"Turn":46.7,"Unassigned":30.1}
print("per-behavior recall:")
for i,c in enumerate(le.classes_):
    print(f"  {c:<12} {cm[i,i]/cm[i].sum()*100:5.1f}%   (ms {ms[c]})")
