import sys, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
df = pd.read_csv('d:/coping-dynamics-sequencing/results/statistical_reports/fig4_verification_results.csv')
trans = df[df['metric'].str.contains('lz|recurrence|determinism|markov', case=False, na=False)]
for _, r in trans.iterrows():
    print(f"{r['metric']}:")
    print(f"  Gold:  beta={r['gold_beta']:.5f} SE={r['gold_se']:.5f} z={r['gold_z']:.3f} p={r['gold_p']:.5f}")
    print(f"  Ours:  beta={r['beta']:.5f} SE={r['se']:.5f} z={r['stat']:.3f} p={r['p']:.5f}")
    print(f"  Tags:  beta={r['beta_tag']} | se={r['se_tag']} | p={r['p_tag']}")
    print()
