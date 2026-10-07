# -*- coding: utf-8 -*-
"""Methodology figure: the evaluation pipeline in five steps.

Requested by Reviewer 2 of ERAMIA-RS 2026 ("the methodology ... is described
mainly in text; another figure describing the steps would make it clearer").
Every number in the boxes is read from the published results or from the run
configuration, not typed from memory; `checa()` asserts them before drawing.

Proportion: the paper builder fits figures to 15 cm wide and at most 4.2 cm
tall, so the aspect ratio must stay at or above 15/4.2 = 3.57 for the figure to
use the full text width.

Usage, from the package root:  python analysis/methodology_figure.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.environ.get("RESULTS_DIR", os.path.join(PKG, "results", "bench_contrato_mi"))
FIG = os.path.join(PKG, "figures"); os.makedirs(FIG, exist_ok=True)


def checa():
    """The figure states the protocol; the published results must agree with it."""
    d = pd.concat([pd.read_csv(f, dtype={"montagem": str, "carimbo": str})
                   for f in glob.glob(os.path.join(RES, "*", "metricas.csv"))])
    escada = d[d.montagem.isin(["2", "4", "8", "16", "32", "64"])]
    assert d.carimbo.nunique() == 1
    assert d.target.nunique() == 30 and d.grupo.nunique() == 6
    assert set(d.n_pool) == {98}
    assert sorted(d.fold.unique()) == [0, 1, 2]
    assert set(d.calib_n) == {48} and set(d.n_calib_alvo) == {48}
    assert (d.banda_lo.unique()[0], d.banda_hi.unique()[0]) == (4.0, 40.0)
    assert set(d.ordem) == {5} and set(d.iqr_k) == {1.5} and set(d.car) == {1}
    assert set(d.fases_treino) == {"mi"} and set(d.escala) == {"microvolts"}
    assert escada.modelo.nunique() == 8
    return d.carimbo.iloc[0]


ETAPAS = [
    ("1  Data", [
        "PhysioNet EEGMMIDB",
        "109 subjects, 64 ch, 160 Hz",
        "imagery runs 4, 8, 12",
        "epochs 0.5–2.5 s",
        "rest, left, right"]),
    ("2  Splits", [
        "30 targets, 6 groups of 5",
        "pool per group: 98 others",
        "target epochs in 3",
        "rotated partitions:",
        "2 calibrate, 1 test"]),
    ("3  Nested montages", [
        "64 ⊃ 32 ⊃ 16 ⊃ 8 ⊃ 4 ⊃ 2",
        "by distance to C3/C4",
        "(8 adds Cz and Fz)",
        "nesting checked;",
        "the run aborts if broken"]),
    ("4  One pipeline", [
        "CAR, 4–40 Hz, µV,",
        "Tukey epoch rejection",
        "8 decoders trained",
        "on the pool, calibrated",
        "on 48 target epochs"]),
    ("5  Two levels", [
        "3-class output read as",
        "L1 detection: rest vs MI,",
        "  balanced accuracy",
        "L2 lateralization: L vs R,",
        "  above 0.5 (1 − FNR)"]),
]
RODAPE = ("Paired Wilcoxon signed-rank tests with the subject as the unit (n = 30), "
          "two-sided α = 0.05; one seed (42).")

AZUL, FUNDO, TEXTO = "#1f4e79", "#eef3f8", "#1a1a1a"
plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans"})


def cabe(fig, ax, textos, larg_max, fs, peso="normal", fs_min=4.4):
    """Largest font size <= fs at which every string fits in `larg_max` (data
    units), measured on the real renderer. Shrinks in 0.1 pt steps."""
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    while fs > fs_min:
        larguras = []
        for s in textos:
            t_ = ax.text(0, 0, s, fontsize=fs, weight=peso)
            bb = t_.get_window_extent(rend)
            (x0_, _), (x1_, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            larguras.append(x1_ - x0_); t_.remove()
        if max(larguras) <= larg_max:
            return fs
        fs = round(fs - 0.1, 2)
    raise ValueError(f"texto nao cabe nem em {fs_min} pt: aumente a largura ou encurte")


def desenha(carimbo):
    fig = plt.figure(figsize=(6.3, 1.72))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 6.3); ax.set_ylim(0, 1.72)
    fig.canvas.draw()
    n, x0, gap = len(ETAPAS), 0.05, 0.13
    larg = (6.3 - 2 * x0 - gap * (n - 1)) / n
    topo, base, pad = 1.62, 0.26, 0.06
    # one size for all titles and one for all bodies, so the boxes read as a set
    fs_tit = cabe(fig, ax, [t for t, _ in ETAPAS], larg - 2 * pad, 6.4, "bold")
    fs_txt = cabe(fig, ax, [ln for _, ls in ETAPAS for ln in ls], larg - 2 * pad, 5.7)
    for i, (titulo, linhas) in enumerate(ETAPAS):
        x = x0 + i * (larg + gap)
        ax.add_patch(FancyBboxPatch((x, base), larg, topo - base,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=FUNDO, ec=AZUL, lw=0.7))
        ax.add_patch(FancyBboxPatch((x, topo - 0.26), larg, 0.26,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=AZUL, ec=AZUL, lw=0.7))
        ax.text(x + pad, topo - 0.13, titulo, color="white", fontsize=fs_tit,
                weight="bold", va="center")
        for j, ln in enumerate(linhas):
            ax.text(x + pad, topo - 0.42 - j * 0.205, ln, color=TEXTO,
                    fontsize=fs_txt, va="center")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + larg + 0.015, (topo + base) / 2),
                                         (x + larg + gap - 0.015, (topo + base) / 2),
                                         arrowstyle="-|>", mutation_scale=7,
                                         color="#555", lw=0.8))
    ax.text(x0, 0.12, RODAPE, fontsize=5.7, color=TEXTO, va="center")
    ax.text(6.3 - x0, 0.12, f"contract stamp {carimbo}", fontsize=5.0,
            color="#777", va="center", ha="right")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"figS1_methodology_en.{ext}"), dpi=400,
                    facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    desenha(checa())
    print("figures/figS1_methodology_en.{png,svg,pdf}")
