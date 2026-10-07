"""
Shallow ConvNet — Setup 1 hierárquico (planning/07)

Baseado em: Schirrmeister et al. (2017) — "Deep learning with convolutional
neural networks for EEG decoding and visualization."

Arquitetura rasa que imita FBCSP via square → log pooling.
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
#  Modelo PyTorch
# ---------------------------------------------------------------------------

class ShallowConvNetEncoder(nn.Module):
    """Blocos convolucionais Shallow ConvNet; saída achatada (batch, feat_size)."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 320,
        n_filters: int = 40,
        kernel_temporal: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples

        self.conv_time = nn.Conv2d(1, n_filters, (1, kernel_temporal), bias=False)
        self.conv_spat = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool_size), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        self.feature_size = self._infer_feature_size()

    def _infer_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            return self.forward(x).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = x ** 2                                      # squaring non-linearity
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))         # safe log
        x = self.dropout(x)
        return x.view(x.size(0), -1)


class ShallowConvNetSetup1HierModel(nn.Module):
    """Encoder Shallow + duas cabeças binárias (motor + lateral) — Setup 1."""

    def __init__(self, n_channels: int = 64, n_samples: int = 320, **enc_kw):
        super().__init__()
        self.encoder = ShallowConvNetEncoder(
            n_channels=n_channels, n_samples=n_samples, **enc_kw
        )
        fs = self.encoder.feature_size
        self.fc_motor = nn.Linear(fs, 2)
        self.fc_lat = nn.Linear(fs, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x)
        return self.fc_motor(f), self.fc_lat(f)


# ---------------------------------------------------------------------------
#  Wrapper sklearn
# ---------------------------------------------------------------------------

class ShallowConvNetSetup1Classifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper para Shallow ConvNet — Setup 1 hierárquico."""

    def __init__(
        self,
        n_filters: int = 40,
        kernel_temporal: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 16,
        patience: int = 20,
        optimizer: str = "adam",
        weight_decay: float = 1e-4,
        filter_band: Optional[Tuple[float, float]] = (4, 40),
        sfreq: float = 160.0,
        val_split: float = 0.15,
    ):
        self.n_filters = n_filters
        self.kernel_temporal = kernel_temporal
        self.pool_size = pool_size
        self.pool_stride = pool_stride
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1, 2])

        self.filter_b = self.filter_a = None
        if self.filter_band is not None:
            nyq = self.sfreq / 2.0
            low, high = self.filter_band[0] / nyq, self.filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(5, [low, high], btype="band")

    # ---------- helpers ----------

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

    def _train_loop(self, X, y, epochs, lr, patience, verbose: bool = True):
        n_val = max(1, int(len(X) * self.val_split))
        idx = np.random.permutation(len(X))
        vi, ti = idx[:n_val], idx[n_val:]

        Xt = torch.FloatTensor(X[ti]).to(self.device)
        yt = torch.LongTensor(y[ti]).to(self.device)
        Xv = torch.FloatTensor(X[vi]).to(self.device)
        yv = torch.LongTensor(y[vi]).to(self.device)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch_size, shuffle=True)
        crit = nn.CrossEntropyLoss()
        opt = self._make_optimizer(self.model.parameters(), lr)

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_imp = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for bx, by in loader:
                opt.zero_grad(set_to_none=True)
                om, ol = self.model(bx)
                loss = self._loss_setup1(om, ol, by, crit)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(bx)
            
            train_loss /= len(ti)

            self.model.eval()
            with torch.no_grad():
                om, ol = self.model(Xv)
                vl = self._loss_setup1(om, ol, yv, crit).item()
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"    [ShallowConvNet] Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {vl:.4f}")

            if vl < best_val:
                best_val = vl
                best_w = copy.deepcopy(self.model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    if verbose:
                        print(f"    [ShallowConvNet] Early stopping na época {epoch+1} com Val Loss: {best_val:.4f}")
                    break

        self.model.load_state_dict(best_w)

    # ---------- API pública ----------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ShallowConvNetSetup1Classifier":
        X = self._preprocess(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        nc, ns = X.shape[1], X.shape[2]
        self.model = ShallowConvNetSetup1HierModel(
            n_channels=nc,
            n_samples=ns,
            n_filters=self.n_filters,
            kernel_temporal=self.kernel_temporal,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            dropout=self.dropout,
        ).to(self.device)
        self._train_loop(X, y, self.epochs, self.lr, self.patience)
        return self

    def continue_training(
        self, X: np.ndarray, y: np.ndarray,
        epochs: int = 80, lr: Optional[float] = None, patience: Optional[int] = None,
    ) -> "ShallowConvNetSetup1Classifier":
        if self.model is None:
            raise RuntimeError("continue_training requires fit() first")
        X = self._preprocess(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        self._train_loop(X, y, epochs, lr or self.lr, patience or self.patience)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt)
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
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt)
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


# ---------------------------------------------------------------------------
#  Fábrica
# ---------------------------------------------------------------------------

def create_shallow_convnet_setup1_model(**kw) -> ShallowConvNetSetup1Classifier:
    """Fábrica para Shallow ConvNet no desenho Setup 1 (duas cabeças)."""
    return ShallowConvNetSetup1Classifier(**kw)


if __name__ == "__main__":
    from core.data_loader import load_subject_setup1_hand_lateral
    from core.metrics import compute_metrics_setup1_hierarchical
    
    print("Testing Shallow ConvNet on Subject 1 (Setup 1 mock)...")
    
    X_train, y_train = load_subject_setup1_hand_lateral(1, phase="real", tmin=0.5, tmax=2.5)
    X_test, y_test = load_subject_setup1_hand_lateral(1, phase="mi", tmin=0.5, tmax=2.5)
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Train Labels: {np.bincount(y_train)}")
    print(f"Test Labels: {np.bincount(y_test)}")
    
    model = create_shallow_convnet_setup1_model(epochs=30)
    print("\n--- Fase 1 (Real) ---")
    model.fit(X_train, y_train)
    
    print("\n--- Fase 2 (MI Fine-tuning) ---")
    model.continue_training(X_test, y_test, epochs=30)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics_setup1_hierarchical(y_test, y_pred)
    
    print(f"\nResults: {metrics}")
