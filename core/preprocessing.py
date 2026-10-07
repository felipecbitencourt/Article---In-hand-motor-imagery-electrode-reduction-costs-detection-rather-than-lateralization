"""
EEG Preprocessing Pipeline

This module provides preprocessing functions for EEG data:
- Bandpass filtering (8-30Hz for mu/beta bands)
- Normalization
- Artifact rejection (optional)
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


# Default filter parameters
DEFAULT_LOW_FREQ = 8.0   # Hz - mu band starts
DEFAULT_HIGH_FREQ = 30.0  # Hz - beta band ends
SFREQ = 160  # PhysioNet sampling frequency


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR).
    
    Subtracts the mean of all channels at each time point,
    removing shared noise from electrode impedance.
    
    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_samples) or (n_channels, n_samples)
        
    Returns
    -------
    X_car : np.ndarray
        Re-referenced data
    """
    if X.ndim == 2:
        mean = X.mean(axis=0, keepdims=True)
        return X - mean
    elif X.ndim == 3:
        mean = X.mean(axis=1, keepdims=True)
        return X - mean
    else:
        raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")


def bandpass_filter(
    X: np.ndarray,
    low_freq: float = DEFAULT_LOW_FREQ,
    high_freq: float = DEFAULT_HIGH_FREQ,
    sfreq: float = SFREQ,
    order: int = 5
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.
    
    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_samples) or (n_channels, n_samples)
    low_freq : float
        Low cutoff frequency in Hz
    high_freq : float
        High cutoff frequency in Hz
    sfreq : float
        Sampling frequency in Hz
    order : int
        Filter order
        
    Returns
    -------
    X_filtered : np.ndarray
        Filtered data, same shape as input
    """
    nyq = sfreq / 2.0
    low = low_freq / nyq
    high = high_freq / nyq
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Handle both 2D and 3D arrays
    if X.ndim == 2:
        return signal.filtfilt(b, a, X, axis=1)
    elif X.ndim == 3:
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_filtered[i] = signal.filtfilt(b, a, X[i], axis=1)
        return X_filtered
    else:
        raise ValueError(f"Expected 2D or 3D array, got {X.ndim}D")


def normalize_channels(
    X: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize each channel independently.
    
    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_samples)
    method : str
        'zscore' - zero mean, unit variance per channel
        'minmax' - scale to [0, 1] per channel
        
    Returns
    -------
    X_norm : np.ndarray
        Normalized data
    """
    X_norm = np.zeros_like(X, dtype=np.float64)
    
    for i in range(X.shape[0]):  # For each epoch
        for j in range(X.shape[1]):  # For each channel
            channel_data = X[i, j, :]
            
            if method == 'zscore':
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 0:
                    X_norm[i, j, :] = (channel_data - mean) / std
                else:
                    X_norm[i, j, :] = channel_data - mean
                    
            elif method == 'minmax':
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                if max_val > min_val:
                    X_norm[i, j, :] = (channel_data - min_val) / (max_val - min_val)
                else:
                    X_norm[i, j, :] = 0.0
            else:
                raise ValueError(f"Unknown method: {method}")
    
    return X_norm


def euclidean_alignment(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Apply Euclidean Alignment (EA) from ABAT algorithm.
    Used to align subjects to a common reference covariance space.

    Ridge regularization (``reg * trace(R_mean)/C * I``) is added to the mean
    covariance before the fractional matrix power. Without it, CAR (and any
    linear transform that removes a mode) leaves R_mean rank-deficient and
    ``fractional_matrix_power(R_mean, -0.5)`` silently returns NaN.

    Parameters
    ----------
    X : np.ndarray
        EEG data, shape (n_epochs, n_channels, n_samples)
    reg : float, default=1e-6
        Relative ridge strength. ``1e-6`` is enough to lift a rank-deficient
        spectrum without noticeably distorting the alignment.

    Returns
    -------
    X_aligned : np.ndarray
        Aligned data, same shape. Falls back to ``X`` unchanged if the
        alignment still produces non-finite values.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array for EA, got {X.ndim}D")

    from scipy.linalg import fractional_matrix_power

    covs = np.matmul(X, np.transpose(X, (0, 2, 1)))
    R_mean = np.mean(covs, axis=0)

    C = R_mean.shape[0]
    ridge = reg * (np.trace(R_mean) / max(C, 1))
    R_mean = R_mean + ridge * np.eye(C, dtype=R_mean.dtype)

    R_inv_half = fractional_matrix_power(R_mean, -0.5).real

    if not np.all(np.isfinite(R_inv_half)):
        return X

    X_aligned = np.einsum('ij,njk->nik', R_inv_half, X)

    if not np.all(np.isfinite(X_aligned)):
        return X

    return X_aligned


def preprocess_epochs(
    X: np.ndarray,
    filter_data: bool = True,
    normalize: bool = False,
    apply_ea: bool = False,
    low_freq: float = DEFAULT_LOW_FREQ,
    high_freq: float = DEFAULT_HIGH_FREQ,
    norm_method: str = 'zscore'
) -> np.ndarray:
    """
    Full preprocessing pipeline for epochs.
    
    Parameters
    ----------
    X : np.ndarray
        Raw epochs, shape (n_epochs, n_channels, n_samples)
    filter_data : bool
        Apply bandpass filter
    normalize : bool
        Apply normalization
    apply_ea : bool
        Apply Euclidean Alignment (ABAT pipeline)
    low_freq, high_freq : float
        Filter cutoff frequencies
    norm_method : str
        Normalization method ('zscore' or 'minmax')
        
    Returns
    -------
    X_preprocessed : np.ndarray
        Preprocessed data
    """
    X_out = X.astype(np.float64)
    
    # Step 1: Common Average Reference (removes shared electrode noise)
    X_out = common_average_reference(X_out)
    
    # Step 2: Bandpass filter
    if filter_data:
        X_out = bandpass_filter(X_out, low_freq=low_freq, high_freq=high_freq)
    
    # Step 3: Euclidean Alignment (ABAT - align domain space)
    if apply_ea:
        X_out = euclidean_alignment(X_out)
    
    # Step 4: Normalization (optional)
    if normalize:
        X_out = normalize_channels(X_out, method=norm_method)
    
    return X_out


def reject_epochs_iqr(
    X: np.ndarray,
    y: np.ndarray,
    k: float = 1.5,
    min_keep_ratio: float = 0.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rejeita épocas-outlier usando o critério IQR sobre a amplitude máxima.

    Diferente de ``reject_bad_epochs``, este método é **adaptativo**:
    o limiar é calculado a partir da própria distribuição de amplitudes
    do conjunto recebido (Q3 + k × IQR), em microvolts.

    Pré-requisito: chamar este função APÓS o pré-processamento (CAR + filtro
    passa-banda). Aplicar em sinal bruto inflaria o limiar por causa de
    drift de baixa frequência e artefatos oculares.

    Parameters
    ----------
    X : np.ndarray
        Épocas em VOLTS, shape (n_epochs, n_channels, n_samples).
    y : np.ndarray
        Labels correspondentes.
    k : float, default 1.5
        Multiplicador do IQR. Tukey clássico = 1.5; mais permissivo = 3.0.
    min_keep_ratio : float, default 0.5
        Trava de segurança: nunca mantém menos que esta fração das épocas.
    verbose : bool
        Imprime quantas foram rejeitadas e o limiar usado.

    Returns
    -------
    X_clean, y_clean : tuple of np.ndarray
    """
    X_uv = X * 1e6
    max_amplitudes = np.max(np.abs(X_uv), axis=(1, 2))

    q1, q3 = np.percentile(max_amplitudes, [25, 75])
    iqr = q3 - q1
    upper = q3 + k * iqr

    good = max_amplitudes < upper
    n_kept = int(np.sum(good))

    # Trava de segurança: se sobrou pouco, mantém os min_keep_ratio melhores
    min_keep = max(int(len(X) * min_keep_ratio), 2)
    if n_kept < min_keep:
        sorted_idx = np.argsort(max_amplitudes)
        good = np.zeros(len(X), dtype=bool)
        good[sorted_idx[:min_keep]] = True
        n_kept = int(np.sum(good))

    n_rejected = len(X) - n_kept
    if verbose and n_rejected > 0:
        print(f"IQR reject: {n_rejected}/{len(X)} épocas (limiar={upper:.1f}µV, "
              f"Q1={q1:.1f}, Q3={q3:.1f})")

    return X[good], y[good]


def reject_bad_epochs(
    X: np.ndarray,
    y: np.ndarray,
    threshold_uv: float = 500.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reject epochs with amplitude exceeding threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Epochs data, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray
        Labels
    threshold_uv : float
        Maximum amplitude threshold in microvolts
        
    Returns
    -------
    X_clean : np.ndarray
        Clean epochs
    y_clean : np.ndarray
        Corresponding labels
    """
    # Convert to microvolts if needed (PhysioNet is in volts)
    X_uv = X * 1e6
    
    # Find epochs where max amplitude exceeds threshold
    max_amplitudes = np.max(np.abs(X_uv), axis=(1, 2))
    good_epochs = max_amplitudes < threshold_uv
    
    n_rejected = np.sum(~good_epochs)
    
    # Safety: never reject more than 50% of epochs (data preservation)
    if n_rejected > len(X) * 0.5:
        # Fall back to keeping the best 50% by amplitude
        sorted_indices = np.argsort(max_amplitudes)
        keep_n = max(len(X) // 2, 2)  # Keep at least 2 epochs
        good_epochs = np.zeros(len(X), dtype=bool)
        good_epochs[sorted_indices[:keep_n]] = True
        n_rejected = np.sum(~good_epochs)
    
    if n_rejected > 0:
        print(f"Rejected {n_rejected}/{len(X)} epochs")
    
    return X[good_epochs], y[good_epochs]


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    from core.data_loader import load_subject
    
    print("Testing preprocessing...")
    X, y = load_subject(1)
    print(f"Original: X.shape = {X.shape}")
    
    X_filtered = bandpass_filter(X)
    print(f"After bandpass (8-30Hz): X.shape = {X_filtered.shape}")
    
    X_norm = normalize_channels(X_filtered)
    print(f"After normalization: mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")
    
    X_clean, y_clean = reject_bad_epochs(X_filtered, y)
    print(f"After rejection: {len(X_clean)} epochs remaining")
