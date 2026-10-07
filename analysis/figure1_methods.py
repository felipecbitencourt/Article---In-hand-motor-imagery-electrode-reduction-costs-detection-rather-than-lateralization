# -*- coding: utf-8 -*-
"""Figure 1 of the revised paper: the method, with the montage ladder as one of its steps.

Replaces the submitted Figure 1 (head + table of channel names). The names now live in
the README; the space goes to the rest of the method, which Reviewer 2 asked to see.
Each step is drawn rather than written: the subjects as a waffle, the partitions as a
bar, the pipeline as a chain of chips, the two levels as a decision tree. Same head and
colours as figures.py; the numbers are checked against the published results first.

Needs data_base_physionet/S001/S001R04.edf for the electrode positions.
Usage, from the package root:  python analysis/figure1_methods.py
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

warnings.filterwarnings("ignore")
PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [PKG, os.path.join(PKG, "modelos"), os.path.join(PKG, "experiments", "hierarquico-8"),
                os.path.join(PKG, "analysis")]
import mne                                                            # noqa: E402
from mne.channels.layout import _find_topomap_coords                  # noqa: E402
from core.data_loader import load_run                                 # noqa: E402
import _bench_montagens as B                                          # noqa: E402
from methodology_figure import checa, AZUL, FUNDO, TEXTO              # noqa: E402

FIG = os.path.join(PKG, "figures")
ESCADA = ["2", "4", "8", "16", "32", "64"]
CORE = {c: cm.viridis(v) for c, v in zip(ESCADA, [0.02, 0.24, 0.44, 0.62, 0.80, 0.96])}
GRUPOS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]   # Okabe-Ito
DECOD = ["#1f4e79", "#2e7d32", "#CC79A7", "#a83232", "#5b2c8d", "#56B4E9", "#e08214", "#D55E00"]
CAL, TESTE, CINZA = "#9ecae1", "#fdae6b", "#bdbdbd"
FS = 5.3
TITULOS = ["1  Data and splits", "2  Nested montages", "3  One pipeline", "4  Two levels"]
RODAPE = ("Paired Wilcoxon signed-rank tests, subject as the unit (n = 30), two-sided α = 0.05, "
          "Holm within each family; one seed (42). Channel lists per rung: see the repository.")


def posicoes():
    raw = load_run(1, 4); mne.datasets.eegbci.standardize(raw)
    nomes = raw.info["ch_names"]
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    xy = _find_topomap_coords(raw.info, picks=range(len(nomes)))
    xy = xy - xy.mean(axis=0)                       # symmetrize, as in figures.py
    esp = np.array([[-1.0, 1.0]])
    par = [int(np.argmin(np.linalg.norm(xy - xy[i] * esp, axis=1))) for i in range(len(xy))]
    xy = (xy + xy[par] * esp) / 2.0
    # 0.95 rather than figures.py's 0.85: the head here is smaller, and spreading the
    # electrodes is what keeps the tightest pair (P7-P5) from touching (gap 0.25 -> ~1.1 pt)
    xy = xy / np.abs(xy).max() * 0.95
    entra = {}
    for c in ESCADA:
        for ch in (B.MONTAGENS[c] or nomes):
            entra.setdefault(ch, c)
    assert len(entra) == 64
    return nomes, xy, entra


def ficha(ax, x, y, w, h, txt, fc="white", ec=AZUL, cor=TEXTO, fs=FS, peso="normal"):
    """A rounded chip centred at (x, y)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.03",
                                fc=fc, ec=ec, lw=0.5, zorder=3))
    ax.text(x, y, txt, fontsize=fs, color=cor, ha="center", va="center", weight=peso, zorder=4)


def seta(ax, a, b, cor="#555"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5, color=cor, lw=0.6, zorder=2))


def bloco_dados(ax, x, w, y0, y1):
    """109 subjects as a waffle: 30 targets, the pool, the 6 excluded."""
    ax.text(x + w / 2, y1 - 0.02, "PhysioNet, 109 subjects", fontsize=FS, color=TEXTO, ha="center", va="top")
    ax.text(x + w / 2, y1 - 0.15, "64 ch · 160 Hz · imagery", fontsize=FS - 0.6, color="#555",
            ha="center", va="top")
    ncol, nlin = 14, 8
    topo_g, base_g = y1 - 0.34, y0 + 0.24
    dx, dy = (w - 0.20) / (ncol - 1), (topo_g - base_g) / (nlin - 1)
    gx = x + 0.10
    tipo = ["alvo"] * 30 + ["pool"] * 73 + ["bad"] * 6
    for k, t in enumerate(tipo):
        px, py = gx + (k % ncol) * dx, topo_g - (k // ncol) * dy
        if t == "bad":
            ax.scatter([px], [py], s=6, marker="x", c="#555", lw=0.6, zorder=3)
        else:
            ax.scatter([px], [py], s=7, c=[CINZA if t == "pool" else AZUL], ec="none", zorder=3)
    yl = y0 + 0.09
    itens = [("30 targets", AZUL, "o"), ("pool", CINZA, "o"), ("excluded", "#555", "x")]
    xl = x + 0.08
    for nome, cor, mk in itens:
        ax.scatter([xl], [yl], s=6, c=cor, marker=mk, lw=0.6, zorder=3)
        ax.text(xl + 0.045, yl, nome, fontsize=FS - 0.6, color=TEXTO, va="center")
        xl += 0.06 + 0.052 * len(nome) * (FS - 0.6) / 6


def bloco_cabeca(fig, ax, x, w, y0, y1, nomes, xy, entra, W, H):
    hb = y1 - y0 - 0.04
    ax_h = fig.add_axes([(x + 0.03) / W, (y0 + 0.02) / H, hb / W, hb / H])
    ax_h.set_aspect("equal"); ax_h.axis("off")
    ax_h.add_patch(Circle((0, 0), 1.0, fc="white", ec="#555", lw=0.8, zorder=1))
    ax_h.add_patch(Polygon([[-0.12, 0.99], [0, 1.15], [0.12, 0.99]], fc="white", ec="#555", lw=0.8))
    for s in (-1, 1):
        ax_h.add_patch(Polygon([[s*0.99, 0.12], [s*1.10, 0.06], [s*1.10, -0.06], [s*0.99, -0.12]],
                               fc="white", ec="#555", lw=0.8))
    ordem = {"2": 6, "4": 5, "8": 4, "16": 3, "32": 2, "64": 1}
    for i, ch in enumerate(nomes):
        c = entra[ch]
        ax_h.scatter(*xy[i], s=4.5, c=[CORE[c]], ec="#222", lw=0.2, zorder=ordem[c] + 1)
    ax_h.set_xlim(-1.16, 1.16); ax_h.set_ylim(-1.16, 1.16)
    kx = x + 0.03 + hb + 0.06
    ax.text(kx, y1 - 0.06, "electrodes", fontsize=FS, color=TEXTO, va="center")
    for k, c in enumerate(ESCADA):                  # two columns of three
        xk, yk = kx + 0.05 + (k // 3) * 0.25, y1 - 0.24 - (k % 3) * 0.16
        ax.scatter([xk], [yk], s=14, c=[CORE[c]], ec="#222", lw=0.3, zorder=3)
        ax.text(xk + 0.08, yk, c, fontsize=FS + 0.2, color=TEXTO, va="center")
    ax.text(kx, y0 + 0.05, "nested: each\ncontains the\ndarker ones", fontsize=FS - 0.4,
            color=TEXTO, va="bottom", linespacing=1.15)


def bloco_pipeline(ax, x, w, y0, y1):
    """Preprocessing chips; pool -> target partitions; the eight decoders."""
    chips = ["CAR", "4–40 Hz", "µV", "Tukey"]
    cw, gap = (w - 0.14 - 3 * 0.05) / 4, 0.05
    yc = y1 - 0.12
    for k, t in enumerate(chips):
        xc = x + 0.07 + cw / 2 + k * (cw + gap)
        ficha(ax, xc, yc, cw, 0.15, t, fs=FS - 0.8)
        if k < 3:
            seta(ax, (xc + cw / 2, yc), (xc + cw / 2 + gap, yc))
    ax.text(x + w / 2, yc - 0.13, "same for every montage", fontsize=FS - 0.6, color="#555",
            ha="center", va="center", style="italic")
    yt = yc - 0.42
    ficha(ax, x + 0.07 + 0.21, yt, 0.42, 0.20, "pool\n98 subj.", fc="#e8eef5", fs=FS - 0.4)
    x_bar = x + 0.07 + 0.42 + 0.10
    lb, hb = (x + w - 0.20 - x_bar - 0.04) / 3, 0.15
    seta(ax, (x + 0.07 + 0.42, yt), (x_bar - 0.01, yt))
    for k, (rot, cor) in enumerate((("cal", CAL), ("cal", CAL), ("test", TESTE))):
        xb = x_bar + k * (lb + 0.02)
        ax.add_patch(Rectangle((xb, yt - hb / 2), lb, hb, fc=cor, ec="white", lw=0.5, zorder=3))
        ax.text(xb + lb / 2, yt, rot, fontsize=FS - 0.8, ha="center", va="center", zorder=4)
    ax.text(x_bar + 3 * (lb + 0.02) + 0.01, yt, "↻3", fontsize=FS + 0.4, va="center", color=TEXTO)
    ax.text(x_bar + 1.5 * lb, yt - 0.14, "target: 48 calibration epochs", fontsize=FS - 1.0,
            color="#555", ha="center", va="center")
    yd = y0 + 0.10
    for k, cor in enumerate(DECOD):
        ax.add_patch(Rectangle((x + 0.07 + k * 0.075, yd - 0.03), 0.06, 0.06, fc=cor, ec="none", zorder=3))
    ax.text(x + 0.07 + 8 * 0.075 + 0.03, yd, "8 decoders", fontsize=FS, color=TEXTO, va="center")


def bloco_niveis(ax, x, w, y0, y1):
    """Decision tree: three-class output -> L1 rest vs MI -> L2 left vs right."""
    cx = x + w / 2 + 0.05
    yr, y1n, y2n = y1 - 0.11, y1 - 0.45, y1 - 0.80
    ficha(ax, cx, yr, 0.50, 0.16, "data", fc="#e8eef5", peso="bold")
    xa, xb, cw = x + 0.47, x + w - 0.40, 0.32
    ficha(ax, xa, y1n, cw, 0.15, "rest", fc=CINZA)
    ficha(ax, xb, y1n, cw, 0.15, "MI", fc="#c6dbef")
    for xx in (xa, xb):
        seta(ax, (cx, yr - 0.08), (xx, y1n + 0.08))
    xl, xr = xb - 0.19, xb + 0.19
    ficha(ax, xl, y2n, 0.30, 0.15, "left", fc="#9ecae1")
    ficha(ax, xr, y2n, 0.30, 0.15, "right", fc="#fdae6b")
    for xx in (xl, xr):
        seta(ax, (xb, y1n - 0.08), (xx, y2n + 0.08))
    ax.text(x + 0.06, y1n, "L1", fontsize=FS + 0.4, weight="bold", color=AZUL, va="center")
    ax.text(x + 0.06, y2n, "L2", fontsize=FS + 0.4, weight="bold", color=AZUL, va="center")
    ax.text(x + 0.06, y0 + 0.13, "L1: balanced accuracy", fontsize=FS - 0.6, color="#555", va="center")
    ax.text(x + 0.06, y0 + 0.03, "L2: above its floor", fontsize=FS - 0.6, color="#555", va="center")


def desenha(carimbo):
    W, H = 6.3, 1.76
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    x0, gap, topo, base, pad = 0.04, 0.12, 1.66, 0.25, 0.06
    larg_cab = 1.80
    larg = (W - 2 * x0 - 3 * gap - larg_cab) / 3
    largs = [larg, larg_cab, larg, larg]
    y0, y1 = base + 0.03, topo - 0.27                      # content area of every box
    nomes, xy, entra = posicoes()
    x = x0
    for i, (titulo, w) in enumerate(zip(TITULOS, largs)):
        ax.add_patch(FancyBboxPatch((x, base), w, topo - base,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=FUNDO, ec=AZUL, lw=0.7))
        ax.add_patch(FancyBboxPatch((x, topo - 0.25), w, 0.25,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=AZUL, ec=AZUL, lw=0.7))
        ax.text(x + pad, topo - 0.125, titulo, color="white", fontsize=6.3, weight="bold", va="center")
        if i == 0:
            bloco_dados(ax, x, w, y0, y1)
        elif i == 1:
            bloco_cabeca(fig, ax, x, w, y0, y1, nomes, xy, entra, W, H)
        elif i == 2:
            bloco_pipeline(ax, x, w, y0, y1)
        else:
            bloco_niveis(ax, x, w, y0, y1)
        if i < len(TITULOS) - 1:
            ym = (topo + base) / 2
            ax.add_patch(FancyArrowPatch((x + w + 0.012, ym), (x + w + gap - 0.012, ym),
                                         arrowstyle="-|>", mutation_scale=7, color="#555", lw=0.8))
        x += w + gap
    ax.text(x0, 0.11, RODAPE, fontsize=5.3, color=TEXTO, va="center")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"fig1_methods_en.{ext}"), dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    desenha(checa())
    print("figures/fig1_methods_en.{png,svg,pdf}")
