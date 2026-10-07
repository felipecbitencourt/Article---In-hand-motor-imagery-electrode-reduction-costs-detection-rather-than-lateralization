"""
PhysioNet EEG Motor Movement/Imagery Dataset Loader

This module provides functions to load and extract epochs from the PhysioNet dataset.
Dataset: https://physionet.org/content/eegmmidb/1.0.0/

Runs of interest for hand motor imagery:
- Run 4, 8, 12: Imagine left fist (T1) or right fist (T2) → labels 0, 1

Runs for fists/feet motor imagery (contexto “não é só mão esq./dir.”):
- Run 6, 10, 14: Imagine both fists (T1) or both feet (T2) → labels 2, 3
"""

import os
import warnings
import numpy as np
import mne
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Literal

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / "data_base_physionet"
SFREQ = 160  # Sampling frequency in Hz
N_CHANNELS = 64
EPOCH_DURATION = 3.0  # seconds after cue
RUNS_MI_HANDS = [4, 8, 12]  # MI lateralized hands (left / right)
RUNS_REAL_HANDS = [3, 7, 11]  # Executed fist movement left/right (Setup 1 / planning 07)
RUNS_MI_FISTS_FEET = [6, 10, 14]  # MI both fists / both feet (PhysioNet task 3/4)

ParadigmType = Literal["hands_lr", "fists_feet"]
Setup1Phase = Literal["real", "mi"]

# Setup 1 (hierárquico): rótulos 0=repouso (T0), 1=mão esquerda (T1), 2=mão direita (T2)

# Labels: hands_lr → 0=left, 1=right | fists_feet → 2=both fists, 3=both feet


def get_subject_path(subject_id: int) -> Path:
    """Get path to subject folder."""
    return DATA_DIR / f"S{subject_id:03d}"


def load_run(subject_id: int, run_id: int) -> mne.io.Raw:
    """
    Load a single run from a subject.
    
    Parameters
    ----------
    subject_id : int
        Subject number (1-109)
    run_id : int
        Run number (1-14)
        
    Returns
    -------
    raw : mne.io.Raw
        Raw EEG data
    """
    subject_path = get_subject_path(subject_id)
    file_name = f"S{subject_id:03d}R{run_id:02d}.edf"
    file_path = subject_path / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Limited.*annotation", category=RuntimeWarning)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw


def extract_epochs_from_run(
    raw: mne.io.Raw,
    tmin: float = 0.0,
    tmax: float = 3.0,
    paradigm: ParadigmType = "hands_lr",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract epochs from a raw run.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations
    tmin : float
        Start time relative to event (seconds)
    tmax : float
        End time relative to event (seconds)
    paradigm : str
        ``hands_lr`` — runs 4/8/12: T1→0 (left), T2→1 (right).
        ``fists_feet`` — runs 6/10/14: T1→2 (both fists), T2→3 (both feet).
        
    Returns
    -------
    X : np.ndarray
        Epochs data, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray
        Labels, shape (n_epochs,)
    """
    # Get events from annotations
    events, _ = mne.events_from_annotations(raw, verbose=False)
    
    epochs_list = []
    labels_list = []
    
    n_samples = int((tmax - tmin) * SFREQ)
    
    for event in events:
        sample_idx = event[0]
        event_code = event[2]
        
        # Annotation codes: 1=T0, 2=T1, 3=T2
        if event_code == 2:
            label = 0 if paradigm == "hands_lr" else 2
        elif event_code == 3:
            label = 1 if paradigm == "hands_lr" else 3
        else:
            continue  # Skip rest (T0)
        
        # Extract epoch
        start_sample = sample_idx + int(tmin * SFREQ)
        end_sample = start_sample + n_samples
        
        if end_sample <= raw.n_times:
            epoch_data = raw.get_data(start=start_sample, stop=end_sample)
            epochs_list.append(epoch_data)
            labels_list.append(label)
    
    if len(epochs_list) == 0:
        return np.array([]), np.array([])
    
    X = np.array(epochs_list)
    y = np.array(labels_list)
    
    return X, y


def extract_epochs_hands_with_rest(
    raw: mne.io.Raw,
    tmin: float = 0.0,
    tmax: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrai épocas T0, T1 e T2 nas runs de punho esquerdo/direito (real ou MI).

    Usa os mesmos códigos de evento que ``extract_epochs_from_run`` após
    ``mne.events_from_annotations``: **1 = T0**, **2 = T1**, **3 = T2**.

    Returns
    -------
    X, y
        ``y`` ∈ {0, 1, 2}: repouso, esquerda, direita.
    """
    events, _ = mne.events_from_annotations(raw, verbose=False)
    epochs_list: List[np.ndarray] = []
    labels_list: List[int] = []
    n_samples = int((tmax - tmin) * SFREQ)

    for event in events:
        sample_idx = event[0]
        event_code = event[2]
        if event_code == 1:
            label = 0
        elif event_code == 2:
            label = 1
        elif event_code == 3:
            label = 2
        else:
            continue

        start_sample = sample_idx + int(tmin * SFREQ)
        end_sample = start_sample + n_samples
        if end_sample <= raw.n_times:
            epoch_data = raw.get_data(start=start_sample, stop=end_sample)
            epochs_list.append(epoch_data)
            labels_list.append(label)

    if len(epochs_list) == 0:
        return np.array([]), np.array([])
    return np.array(epochs_list), np.array(labels_list, dtype=np.int64)


def load_subject_setup1_hand_lateral(
    subject_id: int,
    phase: Setup1Phase,
    tmin: float = 0.0,
    tmax: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dados para o Setup 1 (planning/07): movimento real (R3,7,11) ou MI (R4,8,12).

    Parameters
    ----------
    phase
        ``"real"`` — runs 3, 7, 11; ``"mi"`` — runs 4, 8, 12.
    """
    runs = RUNS_REAL_HANDS if phase == "real" else RUNS_MI_HANDS
    parts_x: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for run_id in runs:
        raw = load_run(subject_id, run_id)
        X, y = extract_epochs_hands_with_rest(raw, tmin=tmin, tmax=tmax)
        if len(X) > 0:
            parts_x.append(X)
            parts_y.append(y)
    if not parts_x:
        raise ValueError(f"No Setup 1 epochs for subject {subject_id} phase={phase}")
    return np.concatenate(parts_x, axis=0), np.concatenate(parts_y, axis=0)


def load_setup1_concat(
    subject_ids: List[int],
    phase: Setup1Phase,
    tmin: float = 0.0,
    tmax: float = 3.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatena ``load_subject_setup1_hand_lateral`` para vários sujeitos."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for sid in subject_ids:
        try:
            X, y = load_subject_setup1_hand_lateral(sid, phase, tmin=tmin, tmax=tmax)
            xs.append(X)
            ys.append(y)
        except Exception as e:
            if verbose:
                print(f"Skip subject {sid} ({phase}): {e}")
    if not xs:
        raise ValueError("No subjects loaded for Setup 1 concat")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_subject(
    subject_id: int,
    runs: Optional[List[int]] = None,
    tmin: float = 0.0,
    tmax: float = 3.0,
    include_feet_mi: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load motor imagery epochs for a subject.
    
    Parameters
    ----------
    subject_id : int
        Subject number (1-109)
    runs : list of int, optional
        Run numbers to load. Default: [4, 8, 12] (MI hands).
        If set, ``include_feet_mi`` is ignored; all listed runs use
        paradigm ``hands_lr`` (compatibilidade com chamadas antigas).
    tmin : float
        Start time relative to event (seconds)
    tmax : float
        End time relative to event (seconds)
    include_feet_mi : bool
        If True (and ``runs is None``), also loads runs 6, 10, 14 with labels
        2=both fists, 3=both feet, concatenated after hands (0/1).
        
    Returns
    -------
    X : np.ndarray
        Epochs data, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray
        Labels: 0=left, 1=right; with ``include_feet_mi``, also 2=fists, 3=feet
    """
    all_epochs = []
    all_labels = []

    if runs is not None:
        run_specs = [(rid, "hands_lr") for rid in runs]
    else:
        run_specs = [(rid, "hands_lr") for rid in RUNS_MI_HANDS]
        if include_feet_mi:
            run_specs.extend((rid, "fists_feet") for rid in RUNS_MI_FISTS_FEET)

    for run_id, paradigm in run_specs:
        raw = load_run(subject_id, run_id)
        X, y = extract_epochs_from_run(raw, tmin=tmin, tmax=tmax, paradigm=paradigm)
        
        if len(X) > 0:
            all_epochs.append(X)
            all_labels.append(y)
    
    if len(all_epochs) == 0:
        raise ValueError(f"No epochs found for subject {subject_id}")
    
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y


def load_all_subjects(
    subject_ids: Optional[List[int]] = None,
    runs: Optional[List[int]] = None,
    tmin: float = 0.0,
    tmax: float = 3.0,
    verbose: bool = True,
    include_feet_mi: bool = False,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load data for multiple subjects.
    
    Parameters
    ----------
    subject_ids : list of int, optional
        Subject numbers to load. Default: all 109 subjects
    runs : list of int, optional
        Run numbers to load. Default: [4, 8, 12] (+ 6,10,14 se include_feet_mi)
    tmin, tmax : float
        Epoch time window
    verbose : bool
        Print progress
    include_feet_mi : bool
        Passed to ``load_subject`` when loading each subject.
        
    Returns
    -------
    data : dict
        Dictionary mapping subject_id -> (X, y)
    """
    if subject_ids is None:
        subject_ids = list(range(1, 110))
    
    data = {}
    
    for i, subject_id in enumerate(subject_ids):
        if verbose:
            print(f"Loading subject {subject_id}/{max(subject_ids)}...", end='\r')
        
        try:
            X, y = load_subject(
                subject_id,
                runs=runs,
                tmin=tmin,
                tmax=tmax,
                include_feet_mi=include_feet_mi,
            )
            data[subject_id] = (X, y)
        except Exception as e:
            print(f"\nError loading subject {subject_id}: {e}")
    
    if verbose:
        print(f"\nLoaded {len(data)} subjects successfully.")
    
    return data


def get_channel_names() -> List[str]:
    """Get list of 64 channel names from first subject."""
    raw = load_run(1, 4)
    return raw.ch_names


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")
    X, y = load_subject(1)
    print(f"Subject 1: X.shape = {X.shape}, y.shape = {y.shape}")
    print(f"Labels distribution: {np.bincount(y)}")
    print(f"Channels: {get_channel_names()[:5]}... (64 total)")
