"""
Evaluation Metrics for EEG Classification

Provides functions to compute:
- Accuracy, Cohen's Kappa, BCA, F1-Score
- Hierarchical metrics for Setup 1 (detection + lateralization)
- Information Transfer Rate (ITR)
- BCI Control Rate
- Confusion matrix (flat for CSV)
"""

import warnings
import numpy as np
from typing import Dict, List, Union
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics_setup1_hierarchical(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    trial_duration_s: float = 2.0,
) -> Dict[str, float]:
    """
    Métricas completas para o Setup 1 (3 classes: 0=repouso, 1=esq, 2=dir).

    Nível 1 (Detecção): binário repouso (0) vs. MI de mão (1∪2).
    Nível 2 (Lateralização): entre trials MI verdadeiros, acerto esq/dir.
    Global: 3 classes completas.

    Parameters
    ----------
    y_true, y_pred : array-like
        Rótulos verdadeiros e preditos (0, 1, 2).
    y_proba : array-like, optional
        Probabilidades preditas shape (N, 3) para AUC-ROC.
        Se None, AUC não é computada.
    trial_duration_s : float
        Duração de um trial em segundos (para ITR). Default 2.0s
        (janela 0.5-2.5s = 2s).
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()

    # Suprimir warnings do sklearn quando folds pequenos não contêm todas as classes
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    warnings.filterwarnings("ignore", message="y_true contains classes not in y_pred")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    out: Dict[str, float] = {}

    # ------------------------------------------------------------------
    #  NÍVEL 1 — Detecção: repouso (0) vs MI de mão (1∪2)
    # ------------------------------------------------------------------
    t1_true = (y_true > 0).astype(np.int64)
    t1_pred = (y_pred > 0).astype(np.int64)

    out["accuracy_detection"] = accuracy_score(t1_true, t1_pred)
    out["kappa_detection"] = cohen_kappa_score(t1_true, t1_pred)
    out["bca_detection"] = balanced_accuracy_score(t1_true, t1_pred)
    out["f1_detection"] = f1_score(t1_true, t1_pred, average="binary", zero_division=0)

    # Sensitivity / Specificity (Nível 1)
    cm1 = confusion_matrix(t1_true, t1_pred, labels=[0, 1])
    tn, fp, fn, tp = cm1.ravel()
    out["sensitivity_detection"] = tp / (tp + fn + 1e-10)
    out["specificity_detection"] = tn / (tn + fp + 1e-10)

    # AUC-ROC para detecção (se probabilidades disponíveis)
    if y_proba is not None:
        try:
            proba_det = np.asarray(y_proba)
            # P(MI) = P(esq) + P(dir) = 1 - P(repouso)
            p_mi = 1.0 - proba_det[:, 0]
            out["auc_roc_detection"] = roc_auc_score(t1_true, p_mi)
        except Exception:
            out["auc_roc_detection"] = float("nan")
    else:
        out["auc_roc_detection"] = float("nan")

    # ------------------------------------------------------------------
    #  NÍVEL 2 — Lateralização condicionada a MI verdadeiro
    # ------------------------------------------------------------------
    mask_mi = y_true > 0
    n_mi = int(np.sum(mask_mi))

    if n_mi == 0:
        out["accuracy_lateralization_given_mi"] = float("nan")
        out["kappa_lateralization_given_mi"] = float("nan")
        out["bca_lateralization_given_mi"] = float("nan")
        out["f1_lateralization_given_mi"] = float("nan")
    else:
        y_true_mi = y_true[mask_mi]
        y_pred_mi = y_pred[mask_mi]

        # Acurácia: predição correta (1 ou 2) entre trials MI
        # Nota: predizer 0 (repouso) num trial MI conta como erro
        correct = np.sum(y_pred_mi == y_true_mi)
        out["accuracy_lateralization_given_mi"] = correct / n_mi

        # Para kappa/BCA/F1 de lateralização, tratamos como binário 1 vs 2
        # mas incluímos predições de 0 como erros (mapeamos 0 → classe incorreta)
        # Abordagem: usar as labels {1, 2} diretamente no sklearn
        out["kappa_lateralization_given_mi"] = cohen_kappa_score(
            y_true_mi, y_pred_mi, labels=[1, 2]
        )
        out["bca_lateralization_given_mi"] = balanced_accuracy_score(
            y_true_mi, y_pred_mi
        )
        out["f1_lateralization_given_mi"] = f1_score(
            y_true_mi, y_pred_mi, average="weighted", labels=[1, 2], zero_division=0
        )

    # ------------------------------------------------------------------
    #  GLOBAL — 3 classes
    # ------------------------------------------------------------------
    out["accuracy_3cls"] = accuracy_score(y_true, y_pred)
    out["kappa_3cls"] = cohen_kappa_score(y_true, y_pred)
    out["bca_3cls"] = balanced_accuracy_score(y_true, y_pred)
    out["f1_3cls_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    out["f1_3cls_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # F1 por classe individual
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    out["f1_class_rest"] = float(f1_per[0])
    out["f1_class_left"] = float(f1_per[1])
    out["f1_class_right"] = float(f1_per[2])

    # ------------------------------------------------------------------
    #  Matriz de confusão 3×3 (flat para CSV)
    # ------------------------------------------------------------------
    cm3 = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    for i in range(3):
        for j in range(3):
            label_true = ["rest", "left", "right"][i]
            label_pred = ["rest", "left", "right"][j]
            out[f"cm_{label_true}_pred_{label_pred}"] = int(cm3[i, j])

    # ------------------------------------------------------------------
    #  ITR (Information Transfer Rate)
    # ------------------------------------------------------------------
    out["itr_detection_bpm"] = compute_itr(
        out["accuracy_detection"], n_classes=2, trial_duration=trial_duration_s
    )
    out["itr_3cls_bpm"] = compute_itr(
        out["accuracy_3cls"], n_classes=3, trial_duration=trial_duration_s
    )

    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute basic classification metrics.
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'bca': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 2
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute comprehensive metrics including confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)

    metrics = compute_metrics(y_true, y_pred)
    metrics.update({
        'confusion_matrix': cm,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'sensitivity_mean': np.mean(sensitivity),
        'specificity_mean': np.mean(specificity),
    })

    return metrics


def compute_itr(
    accuracy: float,
    n_classes: int = 2,
    trial_duration: float = 5.0
) -> float:
    """
    Compute Information Transfer Rate (ITR) in bits per minute.

    ITR = (log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))) * (60/T)

    Parameters
    ----------
    accuracy : float
        Classification accuracy (0-1)
    n_classes : int
        Number of classes
    trial_duration : float
        Duration of one trial in seconds

    Returns
    -------
    itr : float
        Information transfer rate in bits/min
    """
    if accuracy <= 1.0 / n_classes:
        return 0.0

    if accuracy >= 1.0:
        accuracy = 0.9999

    p = accuracy
    n = n_classes

    bits = np.log2(n) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    trials_per_minute = 60.0 / trial_duration
    itr = bits * trials_per_minute

    return max(0.0, float(itr))


def bci_control_rate(
    accuracies: np.ndarray,
    threshold: float = 0.70
) -> float:
    """
    Compute percentage of subjects achieving BCI control.
    A subject has BCI control if accuracy >= threshold.

    Returns percentage (0-100).
    """
    accuracies = np.asarray(accuracies)
    if len(accuracies) == 0:
        return 0.0
    n_control = np.sum(accuracies >= threshold)
    return 100.0 * n_control / len(accuracies)


def count_model_parameters(model) -> Dict[str, int]:
    """
    Count number of parameters in a model.

    Supports:
    - PyTorch nn.Module (via .parameters())
    - scikit-learn estimators (returns 0 — not applicable)
    - Wrappers with .model attribute pointing to PyTorch module

    Returns
    -------
    dict with 'total_params' and 'trainable_params'
    """
    try:
        import torch.nn as nn

        # Try direct PyTorch module
        pytorch_model = None
        if isinstance(model, nn.Module):
            pytorch_model = model
        elif hasattr(model, "model") and isinstance(model.model, nn.Module):
            pytorch_model = model.model
        elif hasattr(model, "net") and isinstance(model.net, nn.Module):
            pytorch_model = model.net

        if pytorch_model is not None:
            total = sum(p.numel() for p in pytorch_model.parameters())
            trainable = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
            return {"total_params": total, "trainable_params": trainable}
    except ImportError:
        pass

    return {"total_params": 0, "trainable_params": 0}


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dictionary as a string for display."""
    lines = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"{name}: {value:.{precision}f}")
    return " | ".join(lines)


if __name__ == "__main__":
    # Quick test
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 1, 0])

    metrics = compute_metrics_setup1_hierarchical(y_true, y_pred)
    print("Setup 1 hierarchical metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:>45s}: {v:.4f}")
        else:
            print(f"  {k:>45s}: {v}")
