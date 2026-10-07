# Paired Wilcoxon signed-rank tests

Subject as the unit (n = 30), two-sided, alpha = 0.05. W is the smaller signed-rank sum; r is the matched-pairs rank-biserial correlation. Holm correction within each family; `sig.` uses the Holm-adjusted p.

## A: all decoders

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 vs 8 electrodes | three-class kappa | all eight | 30 | +0.081 | 13 | 452 | 13 | 1.6e-07 | 9.8e-07 | +0.94 | yes |
| 8 vs 2 electrodes | three-class kappa | all eight | 30 | +0.112 | 12 | 453 | 12 | 1.3e-07 | 9.1e-07 | +0.95 | yes |
| 64 vs 8 electrodes | three-class balanced accuracy | all eight | 30 | +0.045 | 20 | 445 | 20 | 6.9e-07 | 2.8e-06 | +0.91 | yes |
| 8 vs 2 electrodes | three-class balanced accuracy | all eight | 30 | +0.079 | 13 | 452 | 13 | 1.6e-07 | 9.8e-07 | +0.94 | yes |
| 64 vs 8 electrodes | detection (balanced accuracy) | all eight | 30 | +0.065 | 11 | 454 | 11 | 1.0e-07 | 8.2e-07 | +0.95 | yes |
| 8 vs 2 electrodes | detection (balanced accuracy) | all eight | 30 | +0.048 | 33 | 432 | 33 | 6.0e-06 | 1.8e-05 | +0.86 | yes |
| 64 vs 8 electrodes | lateralization above floor | all eight | 30 | +0.002 | 179 | 286 | 179 | 0.28 | 0.28 | +0.23 | no |
| 8 vs 2 electrodes | lateralization above floor | all eight | 30 | +0.070 | 45 | 420 | 45 | 3.0e-05 | 6.1e-05 | +0.81 | yes |

## B: per decoder

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 vs 8 electrodes | three-class kappa | CSP+LDA | 30 | +0.045 | 163 | 302 | 163 | 0.16 | 0.32 | +0.30 | no |
| 64 vs 8 electrodes | three-class kappa | FBCSP+SVM | 30 | +0.106 | 61 | 404 | 61 | 1.9e-04 | 9.4e-04 | +0.74 | yes |
| 64 vs 8 electrodes | three-class kappa | Riemannian | 30 | +0.099 | 62 | 403 | 62 | 2.1e-04 | 9.4e-04 | +0.73 | yes |
| 64 vs 8 electrodes | three-class kappa | EEGNet | 30 | +0.130 | 49 | 416 | 49 | 5.0e-05 | 4.0e-04 | +0.79 | yes |
| 64 vs 8 electrodes | three-class kappa | EEGSym | 30 | +0.088 | 93 | 372 | 93 | 0.0032 | 0.0097 | +0.60 | yes |
| 64 vs 8 electrodes | three-class kappa | ShallowConvNet | 30 | +0.079 | 57 | 408 | 57 | 1.2e-04 | 8.6e-04 | +0.75 | yes |
| 64 vs 8 electrodes | three-class kappa | Adaptive Deep CNN | 30 | +0.089 | 57 | 408 | 57 | 1.2e-04 | 8.6e-04 | +0.75 | yes |
| 64 vs 8 electrodes | three-class kappa | STIA-Net | 30 | +0.009 | 201 | 264 | 201 | 0.53 | 0.53 | +0.14 | no |

## C: above floor at 8

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lateralization vs 0 | lateralization above floor | CSP+LDA | 30 | +0.061 | 68 | 397 | 68 | 3.8e-04 | 0.0015 | +0.71 | yes |
| lateralization vs 0 | lateralization above floor | FBCSP+SVM | 30 | +0.035 | 193 | 272 | 193 | 0.42 | 0.42 | +0.17 | no |
| lateralization vs 0 | lateralization above floor | Riemannian | 30 | +0.070 | 96 | 369 | 96 | 0.004 | 0.012 | +0.59 | yes |
| lateralization vs 0 | lateralization above floor | EEGNet | 30 | +0.115 | 31 | 434 | 31 | 3.4e-05 | 1.7e-04 | +0.87 | yes |
| lateralization vs 0 | lateralization above floor | EEGSym | 30 | +0.136 | 27 | 438 | 27 | 2.3e-06 | 1.6e-05 | +0.88 | yes |
| lateralization vs 0 | lateralization above floor | ShallowConvNet | 30 | +0.094 | 29 | 436 | 29 | 3.2e-06 | 1.9e-05 | +0.88 | yes |
| lateralization vs 0 | lateralization above floor | Adaptive Deep CNN | 30 | +0.148 | 20 | 445 | 20 | 6.9e-07 | 5.5e-06 | +0.91 | yes |
| lateralization vs 0 | lateralization above floor | STIA-Net | 30 | -0.013 | 161.5 | 161.5 | 303.5 | 0.14 | 0.29 | -0.31 | no |

## D: leader at 8 (three-class kappa)

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Riemannian vs CSP+LDA | three-class kappa | CSP+LDA | 30 | +0.069 | 109 | 356 | 109 | 0.0099 | 0.05 | +0.53 | yes |
| Riemannian vs FBCSP+SVM | three-class kappa | FBCSP+SVM | 30 | +0.098 | 51 | 414 | 51 | 6.3e-05 | 3.8e-04 | +0.78 | yes |
| Riemannian vs EEGNet | three-class kappa | EEGNet | 30 | +0.025 | 200 | 265 | 200 | 0.52 | 1 | +0.14 | no |
| Riemannian vs EEGSym | three-class kappa | EEGSym | 30 | +0.017 | 196 | 269 | 196 | 0.46 | 1 | +0.16 | no |
| Riemannian vs ShallowConvNet | three-class kappa | ShallowConvNet | 30 | +0.007 | 231 | 234 | 231 | 0.98 | 1 | +0.01 | no |
| Riemannian vs Adaptive Deep CNN | three-class kappa | Adaptive Deep CNN | 30 | +0.027 | 188 | 277 | 188 | 0.37 | 1 | +0.19 | no |
| Riemannian vs STIA-Net | three-class kappa | STIA-Net | 30 | +0.246 | 21 | 444 | 21 | 8.3e-07 | 5.8e-06 | +0.91 | yes |

## D: leader at 8 (detection (balanced accuracy))

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Riemannian vs CSP+LDA | detection (balanced accuracy) | CSP+LDA | 30 | +0.051 | 94 | 371 | 94 | 0.0035 | 0.014 | +0.60 | yes |
| Riemannian vs FBCSP+SVM | detection (balanced accuracy) | FBCSP+SVM | 30 | +0.059 | 68 | 397 | 68 | 3.8e-04 | 0.0023 | +0.71 | yes |
| Riemannian vs EEGNet | detection (balanced accuracy) | EEGNet | 30 | +0.049 | 107 | 358 | 107 | 0.0087 | 0.017 | +0.54 | yes |
| Riemannian vs EEGSym | detection (balanced accuracy) | EEGSym | 30 | +0.054 | 87 | 378 | 87 | 0.002 | 0.01 | +0.63 | yes |
| Riemannian vs ShallowConvNet | detection (balanced accuracy) | ShallowConvNet | 30 | +0.025 | 163 | 302 | 163 | 0.15 | 0.15 | +0.30 | no |
| Riemannian vs Adaptive Deep CNN | detection (balanced accuracy) | Adaptive Deep CNN | 30 | +0.062 | 85 | 350 | 85 | 0.0042 | 0.014 | +0.61 | yes |
| Riemannian vs STIA-Net | detection (balanced accuracy) | STIA-Net | 30 | +0.150 | 34 | 431 | 34 | 6.9e-06 | 4.8e-05 | +0.85 | yes |

## D: leader at 8 (lateralization above floor)

| contrast | metric | decoder | n | mean diff | W | W+ | W- | p | p (Holm) | r | sig. |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Adaptive Deep CNN vs CSP+LDA | lateralization above floor | CSP+LDA | 30 | +0.086 | 38 | 427 | 38 | 1.2e-05 | 7.3e-05 | +0.84 | yes |
| Adaptive Deep CNN vs FBCSP+SVM | lateralization above floor | FBCSP+SVM | 30 | +0.112 | 40 | 425 | 40 | 1.6e-05 | 8.0e-05 | +0.83 | yes |
| Adaptive Deep CNN vs Riemannian | lateralization above floor | Riemannian | 30 | +0.077 | 41 | 424 | 41 | 1.8e-05 | 8.0e-05 | +0.82 | yes |
| Adaptive Deep CNN vs EEGNet | lateralization above floor | EEGNet | 30 | +0.032 | 126 | 339 | 126 | 0.028 | 0.055 | +0.46 | no |
| Adaptive Deep CNN vs EEGSym | lateralization above floor | EEGSym | 30 | +0.011 | 191 | 274 | 191 | 0.4 | 0.4 | +0.18 | no |
| Adaptive Deep CNN vs ShallowConvNet | lateralization above floor | ShallowConvNet | 30 | +0.053 | 84 | 381 | 84 | 0.0016 | 0.0047 | +0.64 | yes |
| Adaptive Deep CNN vs STIA-Net | lateralization above floor | STIA-Net | 30 | +0.160 | 18 | 447 | 18 | 1.0e-05 | 7.2e-05 | +0.92 | yes |

