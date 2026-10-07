# -*- coding: utf-8 -*-
"""Download the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB 1.0.0).

Writes data_base_physionet/S001/S001R04.edf (and the .event files) in the layout
core/data_loader.py expects. Only the hand runs are fetched by default:
imagery 4, 8, 12 (used by the paper) and execution 3, 7, 11 (the two-phase arm
of the benchmark). 654 recordings, about 1.7 GB, for all 109 subjects. Files already present with
the right size are skipped, so the script can be re-run after an interruption.

Source: Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C28G6P (Open Data Commons Attribution v1.0).

Usage:
    python download_physionet.py                  # all subjects, runs 3 4 7 8 11 12
    python download_physionet.py --subjects 1 10  # a range, for a quick test
    python download_physionet.py --runs 4 8 12    # imagery only
"""
import argparse
import time
import urllib.request
from pathlib import Path

URL = "https://physionet.org/files/eegmmidb/1.0.0/S{s:03d}/S{s:03d}R{r:02d}.{ext}"
DEST = Path(__file__).resolve().parent / "data_base_physionet"


def tamanho_remoto(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", -1))


def baixa(url, destino, tentativas=4):
    esperado = tamanho_remoto(url)
    if destino.exists() and destino.stat().st_size == esperado:
        return "ok (present)"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    for k in range(1, tentativas + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            if esperado > 0 and tmp.stat().st_size != esperado:
                raise IOError(f"size {tmp.stat().st_size} != {esperado}")
            tmp.replace(destino)
            return "downloaded"
        except Exception as e:                     # network hiccups: back off, retry
            if k == tentativas:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(2 * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs=2, default=(1, 109), metavar=("FIRST", "LAST"))
    ap.add_argument("--runs", type=int, nargs="+", default=[3, 4, 7, 8, 11, 12])
    a = ap.parse_args()
    for s in range(a.subjects[0], a.subjects[1] + 1):
        for r in a.runs:
            for ext in ("edf", "edf.event"):
                dest = DEST / f"S{s:03d}" / f"S{s:03d}R{r:02d}.{ext}"
                print(f"S{s:03d}R{r:02d}.{ext}: {baixa(URL.format(s=s, r=r, ext=ext), dest)}",
                      flush=True)


if __name__ == "__main__":
    main()
