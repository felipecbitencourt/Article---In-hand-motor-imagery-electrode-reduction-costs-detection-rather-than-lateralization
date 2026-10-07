"""
STIA-Net — Setup 1 hierárquico (planning/07)

Baseado em: Ma et al. (2022) — "A Spatio-Temporal Interactive Attention Network
for Motor Imagery EEG Decoding." IEEE ICSPCC 2022.

GCN (PLV adjacency) || TCN (dilatada) + Atenção multi-head → Fusão → FC.
Inclui wrapper sklearn-compatible com fit / continue_training / predict.
"""

from __future__ import annotations

import os
import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(os.cpu_count())
try:
    torch.set_num_interop_threads(os.cpu_count())
except RuntimeError:
    pass

from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import signal

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from core.preprocessing import common_average_reference


# ---------------------------------------------------------------------------
#  Utilidade: Phase Locking Value (PLV)
# ---------------------------------------------------------------------------

def compute_plv_matrix(X: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz de adjacência PLV média a partir dos dados EEG.

    Parameters
    ----------
    X : np.ndarray, shape (n_epochs, n_channels, n_samples)

    Returns
    -------
    A : np.ndarray, shape (n_channels, n_channels)
    """
    from scipy.signal import hilbert

    n_epochs, n_ch, n_samples = X.shape
    plv_sum = np.zeros((n_ch, n_ch))

    for ep in range(n_epochs):
        analytic = hilbert(X[ep], axis=1)
        phase = np.angle(analytic)
        for m in range(n_ch):
            for n in range(m + 1, n_ch):
                diff = phase[m] - phase[n]
                plv = np.abs(np.mean(np.exp(1j * diff)))
                plv_sum[m, n] += plv
                plv_sum[n, m] += plv

    A = plv_sum / n_epochs
    np.fill_diagonal(A, 1.0)
    return A


def compute_normalized_laplacian(A: np.ndarray) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} a partir de adjacência A."""
    A_hat = A + np.eye(A.shape[0])
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-8)))
    L_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.FloatTensor(L_norm)


# ---------------------------------------------------------------------------
#  Modelo PyTorch
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Uma camada de Graph Convolution: Z = σ(L_norm · X · Θ)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.theta = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, L_norm: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_ch, in_features); L_norm: (n_ch, n_ch)
        out = torch.matmul(L_norm, x)      # (batch, n_ch, in_features)
        out = self.theta(out)               # (batch, n_ch, out_features)
        return out


class STIANetEncoder(nn.Module):
    """
    Encoder STIA-Net: GCN (spatial) || Attention + TCN (temporal) → fusão.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 320,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        tcn_filters: Tuple[int, ...] = (16, 32, 64),
        n_heads: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # --- Spatial path: GCN ---
        self.gcn1 = GCNLayer(n_samples, gcn_hidden)
        self.gcn2 = GCNLayer(gcn_hidden, gcn_out)

        # --- Spatial attention branch ---
        self.gcn_attn = GCNLayer(n_samples, 1)

        # --- Multi-head self-attention ---
        # Embed dim must be divisible by n_heads
        attn_dim = n_samples
        actual_heads = n_heads
        while attn_dim % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.mhsa = nn.MultiheadAttention(
            embed_dim=attn_dim, num_heads=actual_heads, batch_first=True, dropout=dropout,
        )

        # --- Fusion weights ---
        self.gamma_s = nn.Parameter(torch.tensor(0.5))
        self.gamma_t = nn.Parameter(torch.tensor(0.5))

        # --- Temporal path: TCN (dilated causal convolutions) ---
        f1, f2, f3 = tcn_filters
        self.tcn1 = nn.Conv1d(n_channels, f1, kernel_size=3, dilation=1, padding=1)
        self.tcn2 = nn.Conv1d(f1, f2, kernel_size=3, dilation=2, padding=2)
        self.tcn3 = nn.Conv1d(f2, f3, kernel_size=3, dilation=4, padding=4)
        self.tcn_bn = nn.BatchNorm1d(f3)
        self.tcn_drop = nn.Dropout(dropout)

        # --- Feature size ---
        self.spatial_feat_size = n_channels * gcn_out
        self.temporal_feat_size = f3 * n_samples
        self.feature_size = self.spatial_feat_size + self.temporal_feat_size

    def forward(self, x: torch.Tensor, L_norm: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        batch_size = x.size(0)

        # --- Spatial path (GCN) ---
        s = F.relu(self.gcn1(x, L_norm))
        s = torch.sigmoid(self.gcn2(s, L_norm))         # (batch, n_ch, gcn_out)

        # --- Spatial attention weights ---
        w_s = torch.sigmoid(self.gcn_attn(x, L_norm))   # (batch, n_ch, 1)
        S_a = w_s * x                                    # (batch, n_ch, n_samples)

        # --- Multi-head self-attention ---
        T_a, _ = self.mhsa(x, x, x)                     # (batch, n_ch, n_samples)

        # --- Combined attention ---
        gs = torch.sigmoid(self.gamma_s)
        gt = 1.0 - gs
        O_a = gt * T_a + gs * S_a                        # (batch, n_ch, n_samples)

        # --- TCN ---
        t = F.relu(self.tcn1(O_a))
        t = F.relu(self.tcn2(t))
        t = F.relu(self.tcn3(t))
        t = self.tcn_bn(t)
        t = self.tcn_drop(t)

        # --- Fusão ---
        s_flat = s.reshape(batch_size, -1)
        t_flat = t.reshape(batch_size, -1)
        return torch.cat([s_flat, t_flat], dim=1)


class STIANetSetup1HierModel(nn.Module):
    """Encoder STIA-Net + duas cabeças binárias (motor + lateral) — Setup 1."""

    def __init__(self, n_channels: int = 64, n_samples: int = 320, **enc_kw):
        super().__init__()
        self.encoder = STIANetEncoder(n_channels=n_channels, n_samples=n_samples, **enc_kw)
        fs = self.encoder.feature_size
        self.fc_motor = nn.Linear(fs, 2)
        self.fc_lat = nn.Linear(fs, 2)

    def forward(self, x: torch.Tensor, L_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x, L_norm)
        return self.fc_motor(f), self.fc_lat(f)


# ---------------------------------------------------------------------------
#  Wrapper sklearn
# ---------------------------------------------------------------------------

class STIANetSetup1Classifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper para STIA-Net — Setup 1 hierárquico."""

    def __init__(
        self,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        tcn_filters: Tuple[int, ...] = (16, 32, 64),
        n_heads: int = 8,
        dropout: float = 0.5,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 16,
        patience: int = 20,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
        filter_band: Optional[Tuple[float, float]] = (8, 30),
        sfreq: float = 160.0,
        val_split: float = 0.15,
    ):
        self.gcn_hidden = gcn_hidden
        self.gcn_out = gcn_out
        self.tcn_filters = tcn_filters
        self.n_heads = n_heads
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.filter_band = filter_band
        self.sfreq = sfreq
        self.val_split = val_split

        self.model = None
        self.L_norm = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2])

        self.filter_b = self.filter_a = None
        if self.filter_band is not None:
            nyq = self.sfreq / 2.0
            low, high = self.filter_band[0] / nyq, self.filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(4, [low, high], btype="band")

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = common_average_reference(X)
        if self.filter_b is not None:
            Xf = np.zeros_like(X)
            for i in range(X.shape[0]):
                Xf[i] = signal.filtfilt(self.filter_b, self.filter_a, X[i], axis=1)
            return Xf
        return X

    @staticmethod
    def _loss_setup1(om, ol, batch_y, crit):
        ym = (batch_y > 0).long()
        loss_m = crit(om, ym)
        mask = batch_y > 0
        if mask.any():
            loss_l = crit(ol[mask], batch_y[mask] - 1)
        else:
            loss_l = loss_m * 0.0
        return loss_m + loss_l

    def _make_optimizer(self, params, lr):
        if self.optimizer.lower() == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        return optim.Adam(params, lr=lr, weight_decay=self.weight_decay)

    def _train_loop(self, X, y, epochs, lr, patience):
        n_val = max(1, int(len(X) * self.val_split))
        idx = np.random.permutation(len(X))
        vi, ti = idx[:n_val], idx[n_val:]

        Xt = torch.FloatTensor(X[ti]).to(self.device)
        yt = torch.LongTensor(y[ti]).to(self.device)
        Xv = torch.FloatTensor(X[vi]).to(self.device)
        yv = torch.LongTensor(y[vi]).to(self.device)

        L = self.L_norm.to(self.device)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        crit = nn.CrossEntropyLoss()
        opt = self._make_optimizer(self.model.parameters(), lr)

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_imp = 0

        for _ in range(epochs):
            self.model.train()
            for bx, by in loader:
                opt.zero_grad(set_to_none=True)
                om, ol = self.model(bx, L)
                loss = self._loss_setup1(om, ol, by, crit)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                om, ol = self.model(Xv, L)
                vl = self._loss_setup1(om, ol, yv, crit).item()
            if vl < best_val:
                best_val = vl
                best_w = copy.deepcopy(self.model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        self.model.load_state_dict(best_w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "STIANetSetup1Classifier":
        X = self._preprocess(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        nc, ns = X.shape[1], X.shape[2]

        # Compute PLV adjacency and normalized Laplacian
        print("  [STIA-Net] Computing PLV adjacency matrix...")
        A = compute_plv_matrix(X)
        self.L_norm = compute_normalized_laplacian(A)

        self.model = STIANetSetup1HierModel(
            n_channels=nc,
            n_samples=ns,
            gcn_hidden=self.gcn_hidden,
            gcn_out=self.gcn_out,
            tcn_filters=self.tcn_filters,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)
        self._train_loop(X, y, self.epochs, self.lr, self.patience)
        return self

    def continue_training(
        self, X: np.ndarray, y: np.ndarray,
        epochs: int = 80, lr: Optional[float] = None, patience: Optional[int] = None,
    ) -> "STIANetSetup1Classifier":
        if self.model is None:
            raise RuntimeError("continue_training requires fit() first")
        X = self._preprocess(X)
        y = np.asarray(y, dtype=np.int64).ravel()

        # Recompute PLV for the MI domain
        print("  [STIA-Net] Recomputing PLV for MI domain...")
        A = compute_plv_matrix(X)
        self.L_norm = compute_normalized_laplacian(A)

        self._train_loop(X, y, epochs, lr or self.lr, patience or self.patience)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        self.model.eval()
        L = self.L_norm.to(self.device)
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt, L)
            pm = om.argmax(dim=1)
            out = torch.zeros(len(pm), dtype=torch.long, device=self.device)
            out[pm == 0] = 0
            mot = pm == 1
            if mot.any():
                out[mot] = 1 + ol[mot].argmax(dim=1)
        return out.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        self.model.eval()
        L = self.L_norm.to(self.device)
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt, L)
            sm_m = torch.softmax(om, dim=1)
            sm_l = torch.softmax(ol, dim=1)
            p0 = sm_m[:, 0]
            p_motor = sm_m[:, 1]
            proba = torch.zeros(len(p0), 3, device=self.device)
            proba[:, 0] = p0
            proba[:, 1] = p_motor * sm_l[:, 0]
            proba[:, 2] = p_motor * sm_l[:, 1]
            proba /= proba.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return proba.cpu().numpy()


def create_stianet_setup1_model(**kw) -> STIANetSetup1Classifier:
    """Fábrica para STIA-Net no desenho Setup 1 (duas cabeças)."""
    return STIANetSetup1Classifier(**kw)
