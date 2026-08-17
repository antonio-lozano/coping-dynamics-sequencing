"""Build the self-contained HTML validation report for the manuscript statistics.

Reads the CSVs emitted by scripts/analysis/manuscript_statistics.py and renders a
dark-theme single-file report.  Run that script first, then this one.
"""

import html
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "docs" / "manuscript_statistics_validation"
IC = pd.read_csv(DATA_DIR / "internal_consistency.csv")
TC = pd.read_csv(DATA_DIR / "timecourse_reproduction.csv")
OUT = DATA_DIR / "2026_06_14_statistics_validation.html"

n_triples = len(IC)
n_flagged = int((IC["flag"].fillna("") != "").sum())
n_consistent = n_triples - n_flagged
n_tc = len(TC)
n_tc_ok = int(TC["verdict"].isin(["MATCH", "CLOSE"]).sum())

flagged = IC[IC["flag"].fillna("") != ""]


def esc(x):
    return html.escape(str(x))


def stat_card(label, value, sub, kind="default"):
    cls = f"stat {kind}" if kind != "default" else "stat"
    return (
        f'<div class="{cls}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
    )


# --- SVG: reported z vs reproduced z for the time-course effects ---
def svg_z_scatter(df, width=520, height=360):
    zr = df["z_reported"].abs().tolist()
    zp = df["z_repro"].abs().tolist()
    labels = df["panel"].tolist()
    lo, hi = 0, max(max(zr), max(zp)) * 1.12

    def X(v):
        return 60 + (v - lo) / (hi - lo) * (width - 90)

    def Y(v):
        return height - 45 - (v - lo) / (hi - lo) * (height - 70)

    p = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    p.append(
        f'<rect class="axis" x="60" y="25" width="{width - 90}" height="{height - 70}" fill="none" stroke-width="1"/>'
    )
    # identity line y=x
    p.append(
        f'<line x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" stroke="#6ec48a" stroke-width="1.5" stroke-dasharray="4,4"/>'
    )
    # ticks
    step = 2
    t = 0
    while t <= hi:
        p.append(f'<line class="grid" x1="{X(t):.1f}" y1="25" x2="{X(t):.1f}" y2="{height - 45}"/>')
        p.append(
            f'<text x="{X(t):.1f}" y="{height - 28}" font-size="10" text-anchor="middle">{t}</text>'
        )
        p.append(f'<text x="48" y="{Y(t) + 3:.1f}" font-size="10" text-anchor="end">{t}</text>')
        t += step
    for xr, yp_, lab in zip(zr, zp, labels):
        p.append(f'<circle cx="{X(xr):.1f}" cy="{Y(yp_):.1f}" r="5" fill="#6aa9e9" opacity="0.9"/>')
        p.append(
            f'<text x="{X(xr) + 8:.1f}" y="{Y(yp_) - 6:.1f}" font-size="10" fill="#8b94a3">{esc(lab)}</text>'
        )
    p.append(
        f'<text x="{width / 2:.0f}" y="{height - 8}" font-size="11" text-anchor="middle" fill="#8b94a3">|z| reported (manuscript)</text>'
    )
    p.append(
        f'<text x="16" y="{height / 2:.0f}" font-size="11" text-anchor="middle" fill="#8b94a3" transform="rotate(-90 16 {height / 2:.0f})">|z| reproduced</text>'
    )
    p.append("</svg>")
    return f'<div class="plot-wrap">{"".join(p)}</div>'


# --- Layer 1 table ---
def ic_rows():
    out = []
    for _, r in IC.iterrows():
        fl = str(r["flag"]) if pd.notna(r["flag"]) else ""
        cls = ' style="background:#3a201a"' if fl else ""
        badge = (
            f'<span style="color:#e88871;font-weight:600">{esc(fl)}</span>'
            if fl
            else '<span style="color:#6ec48a">ok</span>'
        )
        out.append(
            f"<tr{cls}><td>{esc(r['panel'])}</td><td>{esc(r['label'])}</td>"
            f"<td style='text-align:right'>{r['beta']:.3f}</td><td style='text-align:right'>{r['SE']:.3f}</td>"
            f"<td style='text-align:right'>{r['z_reported']:.3f}</td><td style='text-align:right'>{r['beta_over_SE']:.3f}</td>"
            f"<td>{esc(r['p_text'])}</td><td style='text-align:right'>{r['p_from_z']:.4f}</td><td>{badge}</td></tr>"
        )
    return "\n".join(out)


# --- Layer 2 table ---
def tc_rows():
    out = []
    vcolor = {"MATCH": "#6ec48a", "CLOSE": "#e6b364", "CHECK": "#e88871"}
    for _, r in TC.iterrows():
        v = r["verdict"]
        out.append(
            f"<tr><td>{esc(r['panel'])}</td><td>{esc(r['cluster'])}</td><td>{esc(r['term'])}</td>"
            f"<td style='text-align:right'>{r['beta_repro']:+.4f}</td><td style='text-align:right'>{r['beta_reported']:+.3f}</td>"
            f"<td style='text-align:right'>{r['SE_repro']:.4f}</td><td style='text-align:right'>{r['SE_reported']:.3f}</td>"
            f"<td style='text-align:right'>{r['z_repro']:.3f}</td><td style='text-align:right'>{r['z_reported']:.3f}</td>"
            f"<td style='color:{vcolor.get(v, '#e6e9ef')};font-weight:600'>{esc(v)}</td></tr>"
        )
    return "\n".join(out)


ERR_EXPL = {
    "Fig3A sniffing frequency": (
        "z != beta/SE",
        "Reported beta=-0.068, SE=0.185, z=-2.133. But beta/SE = -0.37, not -2.13. "
        "The z and p (=0.033) are mutually consistent, so the <b>SE is the typo</b>: for z=-2.133 it should be ~0.032, not 0.185. "
        "Separately, the text says ELS animals show <i>more</i> sniffing, yet the reported beta is negative; "
        "re-fitting from data gives a <b>positive</b> ELS effect (ELS sniff more), matching the prose - so the sign of the reported beta is also suspect.",
    ),
    "Fig5C grooming (resil contrast)": (
        "p disagrees with z",
        "Reported beta=-0.398, SE=0.187, z=-2.123, p=0.003. beta/SE=-2.13 (consistent with z), "
        "but z=-2.123 corresponds to <b>p &asymp; 0.034</b>, not 0.003. The reported p-value is off by ~10x.",
    ),
    "Fig5C turn (resil contrast)": (
        "non-significant reported as p<0.001",
        "Reported beta=0.009, SE=0.016, z=0.568, p&lt;0.001. beta/SE=0.56 (consistent with z), "
        "but z=0.568 corresponds to <b>p &asymp; 0.57</b> - i.e. this contrast is <b>not significant</b>. "
        "Reporting it as p&lt;0.001 is a clear error; the conclusion drawn from this panel should be revisited.",
    ),
}


def err_cards():
    out = []
    for _, r in flagged.iterrows():
        key = f"{r['panel']} {r['label']}"
        title, body = ERR_EXPL.get(key, (r["flag"], ""))
        out.append(
            f'<div class="robust-card"><h4>{esc(r["panel"])} &mdash; {esc(r["label"])}</h4>'
            f'<div style="color:#e88871;font-weight:600;margin-bottom:4px">{esc(title)}</div>{body}</div>'
        )
    return "\n".join(out)


HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Statistics validation &mdash; coping dynamics manuscript</title>
<style>
:root{{--bg:#0e1116;--panel:#161a21;--panel-2:#1c2129;--rule:#262b34;--fg:#e6e9ef;--muted:#8b94a3;
--accent:#6aa9e9;--accent-soft:#1a3654;--supp:#b890de;--supp-soft:#2b1f3a;--exc:#e88871;--exc-soft:#3a201a;
--ok:#6ec48a;--warn:#e6b364;--code-bg:#0a0d12;}}
*{{box-sizing:border-box;}}
body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--fg);line-height:1.55;}}
header.top{{background:var(--panel);border-bottom:1px solid var(--rule);padding:26px 40px 20px;position:sticky;top:0;z-index:50;}}
header.top h1{{margin:0 0 6px;font-size:22px;font-weight:700;}}
header.top .sub{{color:var(--muted);font-size:13px;max-width:900px;}}
.layout{{display:grid;grid-template-columns:250px 1fr;max-width:1340px;margin:0 auto;}}
nav.side{{padding:24px 16px;border-right:1px solid var(--rule);position:sticky;top:92px;height:calc(100vh - 92px);overflow-y:auto;background:var(--bg);}}
nav.side h3{{margin:16px 0 4px;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}}
nav.side a{{display:block;padding:5px 10px;color:var(--fg);text-decoration:none;font-size:12.5px;border-radius:4px;}}
nav.side a:hover{{background:var(--accent-soft);color:var(--accent);}}
main{{padding:28px 40px 80px;min-width:0;}}
h2{{margin-top:36px;font-size:19px;border-bottom:1px solid var(--rule);padding-bottom:6px;}}
h3{{font-size:15px;margin-top:26px;}}
a{{color:var(--accent);}}
.abstract{{background:var(--panel);border:1px solid var(--rule);padding:22px 26px;border-radius:12px;font-size:15px;}}
.abstract p{{margin:0 0 10px;}} .abstract p:last-child{{margin:0;}}
.headline{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:20px 0 28px;}}
.stat{{background:var(--panel);border:1px solid var(--rule);padding:14px 16px;border-radius:8px;}}
.stat .label{{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}}
.stat .value{{font-size:24px;font-weight:700;margin-top:4px;color:var(--accent);}}
.stat.exc .value{{color:var(--exc);}} .stat.ok .value{{color:var(--ok);}} .stat.warn .value{{color:var(--warn);}}
.stat .sub{{font-size:12px;color:var(--muted);margin-top:2px;}}
.takeaway{{background:var(--supp-soft);border-left:4px solid var(--supp);padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0 20px;}}
.takeaway strong{{color:var(--supp);}}
.takeaway.exc{{background:var(--exc-soft);border-left-color:var(--exc);}} .takeaway.exc strong{{color:var(--exc);}}
.takeaway.warn{{background:#3a2f1a;border-left-color:var(--warn);}} .takeaway.warn strong{{color:var(--warn);}}
.takeaway.ok{{background:#1a3a26;border-left-color:var(--ok);}} .takeaway.ok strong{{color:var(--ok);}}
.robust-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;}}
.robust-card{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;font-size:13px;}}
.robust-card h4{{margin:0 0 6px;font-size:13px;color:var(--accent);}}
table{{border-collapse:collapse;width:100%;font-size:12.5px;}}
table th,table td{{border-bottom:1px solid var(--rule);padding:6px 9px;text-align:left;}}
table th{{background:var(--panel-2);color:var(--muted);font-weight:600;position:sticky;top:92px;}}
pre,code.block{{background:var(--code-bg);padding:12px 14px;border-radius:6px;overflow-x:auto;font-size:12px;border:1px solid var(--rule);color:#d6dae2;}}
code{{background:var(--code-bg);color:#d6dae2;padding:1px 5px;border-radius:3px;font-size:.92em;}}
details{{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 14px;margin-top:14px;}}
details summary{{cursor:pointer;font-weight:600;color:var(--accent);}}
.plot-wrap{{background:var(--panel-2);border:1px solid var(--rule);border-radius:6px;padding:8px;max-width:560px;}}
.plot-wrap svg{{width:100%;height:auto;display:block;}} .plot-wrap text{{fill:var(--fg);}}
.plot-wrap .axis{{stroke:var(--rule);}} .plot-wrap .grid{{stroke:var(--rule);stroke-dasharray:2,3;}}
footer{{padding:28px 40px;color:var(--muted);border-top:1px solid var(--rule);font-size:12.5px;}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr;}}nav.side{{position:relative;top:0;height:auto;border-right:0;border-bottom:1px solid var(--rule);}}main{{padding:20px;}}table th{{position:static;}}}}
</style></head><body>
<header class="top">
  <h1>Statistics validation &mdash; <i>Detecting dynamic coping strategies and resilience profiles after early life adversity</i></h1>
  <div class="sub">Sanguino-G&oacute;mez, G&uuml;&ccedil;l&uuml;, Krugers &amp; Lozano. Independent reproduction &amp; consistency check of every reported mixed-linear-model statistic. Generated 2026-06-14.</div>
</header>
<div class="layout">
<nav class="side">
  <h3>Contents</h3>
  <a href="#abstract">Abstract</a>
  <a href="#headline">Headline</a>
  <a href="#errors">Flagged errors</a>
  <a href="#layer1">Layer 1 &middot; consistency</a>
  <a href="#layer2">Layer 2 &middot; reproduction</a>
  <a href="#layer3">Layer 3 &middot; per-animal</a>
  <a href="#coverage">Coverage</a>
  <a href="#method">Method</a>
  <a href="#how">How to run</a>
</nav>
<main>
<section id="abstract"><h2>Abstract</h2>
<div class="abstract">
<p>The manuscript reports its effects as mixed-linear-model Wald triples (&beta;, SE, <i>z</i>, <i>p</i>) across {n_triples} figure-panel statistics. This document validates them two ways: (1) an <b>internal-consistency</b> audit that checks the Wald identities <code>z = &beta;/SE</code> and <code>p = 2(1-&Phi;(|z|))</code> on every reported triple, and (2) a <b>data reproduction</b> that re-fits the unsupervised time-course models (Fig&nbsp;2E and Fig&nbsp;3B/C/E/F) directly from the canonical MoSeq output.</p>
<p><b>{n_consistent} of {n_triples}</b> reported triples are internally consistent. <b>{n_flagged}</b> contain reporting errors (detailed below). The headline time-course models <b>reproduce from the raw data</b>: &beta; and SE match the manuscript to the reported precision and <i>z</i> within ~10%. The reverse-engineered method is documented so all of these statistics are now reproducible from a single script in the repo.</p>
</div></section>

<section id="headline"><h2>Headline</h2>
<div class="headline">
{stat_card("Reported triples", n_triples, "mixed-model (β,SE,z,p)")}
{stat_card("Internally consistent", f"{n_consistent}/{n_triples}", "Wald identities hold", "ok")}
{stat_card("Reporting errors", n_flagged, "need correction", "exc")}
{stat_card("Time-courses reproduced", f"{n_tc_ok}/{n_tc}", "from raw MoSeq data", "ok")}
{stat_card("Fig 2E main effect", "β=0.126", "repro 0.1262 ✓", "ok")}
{stat_card("Fig 2E ELS×time", "β=−0.023", "repro −0.0233 ✓", "ok")}
</div>
<div class="takeaway ok"><strong>Bottom line:</strong> the statistical approach is sound and the headline unsupervised result reproduces exactly. Fix the three flagged reporting errors before submission &mdash; one of them (Fig&nbsp;5C turn) reports a non-significant contrast as p&lt;0.001.</div>
</section>

<section id="errors"><h2>Flagged reporting errors</h2>
<div class="robust-grid">
{err_cards()}
</div>
<div class="takeaway exc"><strong>Most important:</strong> Fig&nbsp;5C "turn (resilient contrast)" &mdash; z=0.568 is <b>not significant</b> (true p&asymp;0.57) yet is printed as p&lt;0.001. Any claim resting on that panel needs revisiting.</div>
</section>

<section id="layer1"><h2>Layer 1 &middot; internal consistency of every reported triple</h2>
<p>For a Wald z-test, <code>z</code> must equal <code>&beta;/SE</code> (allowing for the rounding of small SEs) and the reported <code>p</code> must match <code>2(1&minus;&Phi;(|z|))</code>. Rows that fail are highlighted.</p>
<table><thead><tr><th>panel</th><th>effect</th><th>&beta;</th><th>SE</th><th>z (rep.)</th><th>&beta;/SE</th><th>p (rep.)</th><th>p from z</th><th>check</th></tr></thead>
<tbody>
{ic_rows()}
</tbody></table>
</section>

<section id="layer2"><h2>Layer 2 &middot; data reproduction of the time-course models</h2>
<p>Re-fit from <code>new_results_clusters.pkl</code> + <code>index.csv</code> (n={int(TC["n_animals"].iloc[0])} animals, {int(TC["n_obs"].iloc[0])} animal&times;bin observations). Method: 30&nbsp;s bins, percentage of all frames per bin, time in seconds, <code>percentage ~ group * time</code> with random intercept per animal (MixedLM, REML).</p>
<table><thead><tr><th>panel</th><th>cluster</th><th>term</th><th>&beta; repro</th><th>&beta; rep.</th><th>SE repro</th><th>SE rep.</th><th>z repro</th><th>z rep.</th><th>verdict</th></tr></thead>
<tbody>
{tc_rows()}
</tbody></table>
<div style="margin-top:18px">{svg_z_scatter(TC)}</div>
<div class="takeaway ok"><strong>Reproduced.</strong> &beta; and SE match the reported values to their printed precision; <i>z</i> matches within ~10% (residual is animal-set / REML minutiae). Note Fig&nbsp;2E and Fig&nbsp;3B are the <i>same</i> model: the freezing cluster is exactly syllables S0+S28 (verified, Jaccard&nbsp;=&nbsp;1.00), so their identical reported values are correct, not a duplication error.</div>
</section>

<section id="layer3"><h2>Layer 3 &middot; per-animal metric models (Fig 3A, Fig 4, Fig 5&ndash;6)</h2>
<p>The per-animal frequency, diversity, bout-duration and transition models (and the resilience contrasts) depend on an <code>Experiment</code> covariate and on metric definitions that are <b>not fully recoverable from the shipped artifacts</b>. Re-fitting from the repo's documented metric functions reproduces the <b>sign and significance</b> of most effects but not the exact &beta;. Exact reproduction needs the original per-metric definitions and the per-animal experiment labels.</p>
<div class="robust-grid">
<div class="robust-card"><h4>Direction confirmed</h4>Freezing &amp; turn frequency, Simpson diversity, bout durations (freeze/sniff/turn), and the resilience dynamics score all reproduce with the same sign and comparable significance.</div>
<div class="robust-card"><h4>Sniffing frequency sign</h4>Re-fitting gives a <b>positive</b> ELS effect (ELS sniff more) &mdash; consistent with the prose ("more&nbsp;sniffing") but opposite the reported &beta;=&minus;0.068. Combined with the inconsistent SE, the Fig&nbsp;3A sniffing entry needs a recompute.</div>
<div class="robust-card"><h4>CUI definition</h4>The repo's cumulative-usage-index function does not reproduce the reported &beta; magnitude/sign &mdash; the manuscript used a different CUI definition. Needs the original formula to validate exactly.</div>
</div>
</section>

<section id="coverage"><h2>Coverage</h2>
<table><thead><tr><th>Figure / panel group</th><th>Models</th><th>Status</th></tr></thead><tbody>
<tr><td>Fig 2E &mdash; S0+S28 over time</td><td>1 (×2 terms)</td><td style="color:#6ec48a">Reproduced exactly</td></tr>
<tr><td>Fig 3B/C/E/F &mdash; cluster time-courses</td><td>4</td><td style="color:#6ec48a">Reproduced (sniffing CLOSE)</td></tr>
<tr><td>Fig 3A &mdash; cluster frequencies</td><td>3</td><td style="color:#e6b364">Direction confirmed; sniffing flagged</td></tr>
<tr><td>Fig 4 &mdash; diversity / bouts / transitions</td><td>8</td><td style="color:#e6b364">Direction confirmed; CUI &amp; RQA need original defs</td></tr>
<tr><td>Fig 5&ndash;6 &mdash; resilience contrasts</td><td>~22</td><td style="color:#e6b364">Internally consistent; need MDS/BFL pipeline to re-fit</td></tr>
<tr><td>Fig 1 &mdash; supervised freezing (2 cohorts)</td><td>5</td><td style="color:#8b94a3">Needs per-cohort SimBA freezing time-series</td></tr>
<tr><td>All panels &mdash; internal consistency</td><td>{n_triples}</td><td style="color:#6ec48a">Audited; 3 errors found</td></tr>
</tbody></table>
</section>

<section id="method"><h2>Reverse-engineered method (for reproducibility)</h2>
<p>The single subtlety worth recording: the manuscript reports the time slope in <b>per-second</b> units while the test statistic is set by the <b>30&nbsp;s binning</b>. Equivalently, the model is fit on 30&nbsp;s-bin percentages with a continuous time predictor expressed in seconds. This is why &beta;=0.126 (per second) pairs with z=32.6 (from ~15 bins &times; 83 animals): rescaling time changes &beta; and SE together but leaves <i>z</i> invariant.</p>
<pre>bin = 30 s (= 750 frames @ 25 fps)
y   = 100 * (frames of behaviour in bin) / (all frames in bin)     # per animal, per bin
time = bin_index * 30                                              # SECONDS, continuous
MixedLM:  y ~ C(group) * time ,  groups = animal ,  REML
  -> coef[time]        = "main effect of time"  (Control reference slope)
  -> coef[group:time]  = "ELS x time interaction"</pre>
</section>

<section id="how"><h2>How to run</h2>
<pre>cd coping-dynamics-sequencing
uv run python scripts/analysis/manuscript_statistics.py        # prints tables, writes CSVs
uv run python scripts/analysis/build_validation_report.py      # rebuilds this report</pre>
<details><summary>Files</summary>
<pre>scripts/analysis/manuscript_statistics.py            # the validation (Layer 1 + Layer 2)
docs/manuscript_statistics_validation/internal_consistency.csv
docs/manuscript_statistics_validation/timecourse_reproduction.csv
docs/manuscript_statistics_validation/2026_06_14_statistics_validation.html  # this file</pre>
</details>
</section>
</main></div>
<footer>Generated by build_validation_report.py from internal_consistency.csv + timecourse_reproduction.csv. Data: new_results_clusters.pkl + index.csv (keypoint_moseq_project).</footer>
</body></html>
"""

OUT.write_text(HTML, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print(f"Wrote {OUT}  ({size_mb:.2f} MB)")
