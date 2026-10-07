# -*- coding: utf-8 -*-
"""Compare a fresh run of the benchmark with the published results.

Every (montage, decoder, target, fold) cell present in the new run is matched to
the published one, and every numeric metric and every per-epoch prediction is
compared. Classical decoders (CSP+LDA, FBCSP+SVM, Riemannian) are deterministic
and should match exactly; deep decoders are seeded but use cuDNN's default
(non-deterministic) kernels, so small differences are expected there.

Usage, from the package root:
    python analysis/compare_runs.py experiments/hierarquico-8/results/my_run/ml
"""
import os
import sys

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PUB = os.path.join(PKG, "results", "bench_contrato_mi")
DT = {"montagem": str, "carimbo": str, "modelo": str}
CHAVE = ["montagem", "modelo", "target", "fold"]


def carrega(nome, pasta):
    return pd.read_csv(os.path.join(pasta, nome), dtype=DT)


def publicados(nome):
    return pd.concat([carrega(nome, os.path.join(PUB, s)) for s in ("ml", "deep")],
                     ignore_index=True)


def main(nova):
    novo, pub = carrega("metricas.csv", nova), publicados("metricas.csv")
    if set(novo.carimbo) != set(pub.carimbo):
        print(f"different contract stamps: new {sorted(set(novo.carimbo))}, "
              f"published {sorted(set(pub.carimbo))} -- not the same experiment")
        return 1
    m = novo.merge(pub, on=CHAVE, suffixes=("_n", "_p"))
    cols = [c for c in novo.columns
            if c not in CHAVE + ["secs", "grupo"] and is_numeric_dtype(novo[c]) and c + "_p" in m]
    print(f"metrics: {len(novo)} new rows, {len(m)} matched to published cells")
    for (mt, md), g in m.groupby(["montagem", "modelo"]):
        dif = max(float(np.nanmax(np.abs(g[c + "_n"].astype(float) - g[c + "_p"].astype(float))))
                  for c in cols)
        print(f"  montage {mt:>3} {md:<16} {len(g):3d} cells  max |diff| over "
              f"{len(cols)} metrics = {dif:.2e}  {'EXACT' if dif == 0 else ''}")
    pn, pp = carrega("predicoes.csv", nova), publicados("predicoes.csv")
    k = CHAVE + ["epoch_idx"]
    mm = pn.merge(pp, on=k, suffixes=("_n", "_p"))
    same = (mm.y_pred_n == mm.y_pred_p).mean() if len(mm) else float("nan")
    dp = max(float(np.abs(mm[f"{c}_n"] - mm[f"{c}_p"]).max())
             for c in ("p_rest", "p_left", "p_right")) if len(mm) else float("nan")
    print(f"predictions: {len(mm)} epochs matched, identical labels {100 * same:.2f}%, "
          f"max |diff| in probabilities {dp:.2e}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
