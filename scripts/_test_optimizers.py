"""
Try different MixedLM fitting options to reproduce SE=0.00659 for Simpson.
The Hessian is not positive definite at the boundary — try different optimizers
and SE computation methods.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams

sys.path.insert(0, 'd:/coping-dynamics-sequencing')
from src.config import SYLLABLE_TIMEBIN_250MS

CLUSTER_MAP = {
    "Freezing": [0, 28], "Sniffing": [18, 20], "Grooming": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27], "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climbing": [111], "Jump": [23, 29, 30, 34],
}
raw = pd.read_csv(SYLLABLE_TIMEBIN_250MS)
raw = raw.rename(columns={"Time Bin": "Time_bin"})
raw = raw[raw["Condition"].isin(["Control", "ELS"])].copy()
raw["Animal"] = raw["Animal"].astype(str)
raw["Syllable"] = pd.to_numeric(raw["Syllable"], errors="coerce").astype(int)
s2c = {}
for c, ss in CLUSTER_MAP.items():
    for s in ss: s2c[int(s)] = c
raw["Cluster"] = raw["Syllable"].map(s2c).fillna("")

freq_df = raw.groupby(['Animal', 'Experiment', 'Condition', 'Cluster']).size().reset_index(name='Count')
pivot = freq_df.pivot_table(index=['Animal','Condition','Experiment'], columns='Cluster', values='Count', fill_value=0)
total = pivot.sum(axis=1)
rel = pivot.div(total, axis=0)
simpson = rel.apply(lambda r: 1 - np.sum(r**2), axis=1)
fm = simpson.reset_index().rename(columns={0: 'Simpson'})

print("Gold: beta=-0.01344087 SE=0.00659293 z=-2.03867875 p=0.04148210")
print()

pk = "Condition[T.ELS]"

# Test different optimizers
for method in ['bfgs', 'lbfgs', 'cg', 'nm', 'powell']:
    try:
        m = smf.mixedlm("Simpson ~ Condition + Experiment", data=fm,
                         groups=fm['Animal']).fit(reml=False, method=method)
        print(f"method={method}: SE={m.bse[pk]:.8f} z={m.tvalues[pk]:.8f} p={m.pvalues[pk]:.8f}")
    except Exception as e:
        print(f"method={method}: ERROR {e}")

# Test: fix random effect variance at a small positive value
print()
print("--- Fixed random effect variance ---")
model = MixedLM.from_formula("Simpson ~ Condition + Experiment", data=fm, groups="Animal")
for re_var in [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003]:
    try:
        # Initialize with fixed re variance
        params0 = model.fit(reml=False).params
        result = model.fit(reml=False, start_params=params0)
        print(f"  re_var={re_var}: SE={result.bse[pk]:.8f} (can't fix easily)")
        break
    except Exception as e:
        print(f"  re_var={re_var}: {e}")
        break

# Test: profile likelihood SE
print()
print("--- Profile likelihood CI (if available) ---")
try:
    m = smf.mixedlm("Simpson ~ Condition + Experiment", data=fm,
                     groups=fm['Animal']).fit(reml=False)
    # Try to get profile likelihood based CI
    ci = m.conf_int()
    beta = m.params[pk]
    ci_lo = ci.loc[pk, 0]
    ci_hi = ci.loc[pk, 1]
    se_from_ci = (ci_hi - ci_lo) / (2 * 1.96)
    print(f"  SE from 95% CI: {se_from_ci:.8f}")
    print(f"  CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
except Exception as e:
    print(f"  Error: {e}")

# Test: OLS residual-based bootstrap SE
print()
print("--- Bootstrap SE (n=1000) ---")
np.random.seed(42)
betas = []
for _ in range(1000):
    boot = fm.sample(n=len(fm), replace=True)
    try:
        mb = smf.ols("Simpson ~ Condition + Experiment", data=boot).fit()
        betas.append(mb.params.get("Condition[T.ELS]", np.nan))
    except:
        pass
boot_se = np.std(betas)
boot_z = np.mean(betas) / boot_se
print(f"  Bootstrap SE={boot_se:.8f} z={np.mean(betas)/boot_se:.6f}")
from scipy.stats import norm
p_boot = 2 * norm.cdf(-abs(boot_z))
print(f"  Bootstrap p={p_boot:.8f}")
