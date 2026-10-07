# -*- coding: utf-8 -*-
"""Paired statistics behind the paper's claims, with the full test report.

Reviewer 1 (ERAMIA-RS 2026) asked for the significance level and the Wilcoxon
statistic W to be reported. Every test here is a Wilcoxon signed-rank test with
the subject as the unit (n = 30 targets), two-sided, alpha = 0.05. A subject's
value is the mean over its three test partitions; "all decoders" means the
subject's mean over the eight decoders. Families of tests across the eight
decoders are corrected with Holm.

Reported for each test: n, the mean and median paired difference, W (the
smaller of the two signed-rank sums, the two-sided convention of scipy), W+
and W- separately, the exact p value when there are no ties or zeros (normal
approximation otherwise, flagged in `method`), the matched-pairs rank-biserial
correlation r = (W+ - W-) / (W+ + W-), and the Holm-adjusted p where there is a
family.

Usage, from the package root:  python analysis/stats.py
Writes results/stats/wilcoxon_tests.csv and results/stats/wilcoxon_tests.md
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.environ.get("RESULTS_DIR", os.path.join(PKG, "results", "bench_contrato_mi"))
OUT = os.path.join(PKG, "results", "stats"); os.makedirs(OUT, exist_ok=True)
ALPHA = 0.05
ESCADA = ["2", "4", "8", "16", "32", "64"]
NOME = {"csp_lda": "CSP+LDA", "fbcsp_svm": "FBCSP+SVM", "riemannian": "Riemannian",
        "eegnet": "EEGNet", "eegsym": "EEGSym", "shallow_convnet": "ShallowConvNet",
        "deepcnn": "Adaptive Deep CNN", "stia_net": "STIA-Net"}
METRICA = {"kappa_3cls": "three-class kappa", "bca_3cls": "three-class balanced accuracy",
           "bca_detection": "detection (balanced accuracy)",
           "lat": "lateralization above floor"}

d = pd.concat([pd.read_csv(f, dtype={"montagem": str, "carimbo": str})
               for f in glob.glob(os.path.join(RES, "*", "metricas.csv"))], ignore_index=True)
assert d.carimbo.nunique() == 1, "results from more than one contract"
d["lat"] = d.accuracy_lateralization_given_mi - 0.5 * d.sensitivity_detection
d = d[d.montagem.isin(ESCADA)]
# subject level: mean over the three test partitions
S = d.groupby(["montagem", "modelo", "target"])[list(METRICA)].mean().reset_index()
TODOS = S.groupby(["montagem", "target"])[list(METRICA)].mean().reset_index()


def teste(a, b):
    """Two-sided paired Wilcoxon of a - b, with the quantities a report needs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    dif = a - b
    nz = dif[dif != 0]
    r = rankdata(np.abs(nz))
    w_mais, w_menos = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    empates = len(np.unique(np.abs(nz))) < len(nz)
    zeros = int((dif == 0).sum())
    metodo = "exact" if (not empates and zeros == 0 and len(dif) <= 50) else "approx"
    res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox",
                   method="exact" if metodo == "exact" else "approx")
    return dict(n=len(dif), mean_diff=dif.mean(), median_diff=float(np.median(dif)),
                W=float(res.statistic), W_plus=w_mais, W_minus=w_menos,
                p=float(res.pvalue), method=metodo,
                r_rb=(w_mais - w_menos) / (w_mais + w_menos) if (w_mais + w_menos) else 0.0)


def holm(ps):
    ps = np.asarray(ps, float)
    ordem = np.argsort(ps)
    adj = np.empty_like(ps)
    corrente = 0.0
    for k, i in enumerate(ordem):
        corrente = max(corrente, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = corrente
    return adj


def par(tab, col, montagem_a, montagem_b, chave="target"):
    A = tab[tab.montagem == montagem_a].set_index(chave)[col]
    B = tab[tab.montagem == montagem_b].set_index(chave)[col]
    A, B = A.align(B, join="inner")
    return A.values, B.values


linhas = []

# Family A: the headline contrasts, every decoder averaged within subject (Holm
# over the eight contrasts of the family).
fam = []
for met in METRICA:
    for ma, mb in (("64", "8"), ("8", "2")):
        a, b = par(TODOS, met, ma, mb)
        fam.append(dict(family="A: all decoders", contrast=f"{ma} vs {mb} electrodes",
                        metric=METRICA[met], decoder="all eight", **teste(a, b)))
for x, ph in zip(fam, holm([x["p"] for x in fam])):
    x["p_holm"] = ph
linhas += fam

# Family B: three-class kappa, 64 vs 8, per decoder (Holm over eight).
fam = []
for m in NOME:
    a, b = par(S[S.modelo == m], "kappa_3cls", "64", "8")
    fam.append(dict(family="B: per decoder", contrast="64 vs 8 electrodes",
                    metric=METRICA["kappa_3cls"], decoder=NOME[m], **teste(a, b)))
for x, ph in zip(fam, holm([x["p"] for x in fam])):
    x["p_holm"] = ph
linhas += fam

# Family C: at 8 electrodes, is each decoder's lateralization above its floor?
fam = []
for m in NOME:
    v = S[(S.modelo == m) & (S.montagem == "8")].sort_values("target")["lat"].values
    fam.append(dict(family="C: above floor at 8", contrast="lateralization vs 0",
                    metric=METRICA["lat"], decoder=NOME[m], **teste(v, np.zeros_like(v))))
for x, ph in zip(fam, holm([x["p"] for x in fam])):
    x["p_holm"] = ph
linhas += fam

# Family D: at 8 electrodes, the leader of each metric against every other decoder.
for met in ("kappa_3cls", "bca_detection", "lat"):
    s8 = S[S.montagem == "8"].pivot_table(index="target", columns="modelo", values=met)
    lider = s8.mean().idxmax()
    fam = []
    for m in NOME:
        if m == lider:
            continue
        fam.append(dict(family=f"D: leader at 8 ({METRICA[met]})",
                        contrast=f"{NOME[lider]} vs {NOME[m]}", metric=METRICA[met],
                        decoder=NOME[m], **teste(s8[lider].values, s8[m].values)))
    for x, ph in zip(fam, holm([x["p"] for x in fam])):
        x["p_holm"] = ph
    linhas += fam

T = pd.DataFrame(linhas)
T["alpha"] = ALPHA
T["significant"] = T["p_holm"] < ALPHA
T.to_csv(os.path.join(OUT, "wilcoxon_tests.csv"), index=False, float_format="%.6g")


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


with open(os.path.join(OUT, "wilcoxon_tests.md"), "w", encoding="utf-8") as f:
    f.write("# Paired Wilcoxon signed-rank tests\n\n"
            f"Subject as the unit (n = 30), two-sided, alpha = {ALPHA}. W is the smaller "
            "signed-rank sum; r is the matched-pairs rank-biserial correlation. "
            "Holm correction within each family; `sig.` uses the Holm-adjusted p.\n\n")
    for fam_, g in T.groupby("family", sort=False):
        f.write(f"## {fam_}\n\n| contrast | metric | decoder | n | mean diff | W | W+ | W- "
                "| p | p (Holm) | r | sig. |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for _, x in g.iterrows():
            ph = x.get("p_holm")
            f.write(f"| {x.contrast} | {x.metric} | {x.decoder} | {x.n} | {x.mean_diff:+.3f} "
                    f"| {x.W:g} | {x.W_plus:g} | {x.W_minus:g} | {fmt_p(x.p)} "
                    f"| {fmt_p(ph) if pd.notna(ph) else '—'} | {x.r_rb:+.2f} "
                    f"| {'yes' if x.significant else 'no'} |\n")
        f.write("\n")
print(f"{len(T)} tests -> results/stats/wilcoxon_tests.csv and .md")
print(T[["family", "contrast", "metric", "decoder", "mean_diff", "W", "p", "p_holm",
         "significant"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
