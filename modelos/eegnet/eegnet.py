"""
EEGNet - Compact CNN for EEG Classification

A compact convolutional neural network for EEG-based brain-computer interfaces.

Reference: EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs
           (Lawhern et al., 2018)

Paper Parameters:
- F1 = 8 (temporal filters)
- D = 2 (depth multiplier for depthwise conv)
- F2 = 16 (pointwise filters)
- Dropout = 0.5
- Kernel size: (1, 64) for temporal, (C, 1) for spatial
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Forçar PyTorch a usar todos os núcleos da CPU
torch.set_num_threads(os.cpu_count())
try:
    torch.set_num_interop_threads(os.cpu_count())
except RuntimeError:
    pass  # Already set by another module
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import signal
from typing import Optional, Tuple

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from core.preprocessing import common_average_reference


class MaxNormConstraint:
    """
    MaxNorm weight constraint used in EEGNet.
    Constraints the weights to a maximum L2 norm along incoming features.
    """
    def __init__(self, max_val=1.0):
        self.max_val = max_val

    def __call__(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                # L2 norm over all dims except output channels (dim 0)
                dims = tuple(range(1, module.weight.dim()))
                if not dims:
                    return
                norm = module.weight.norm(2, dim=dims, keepdim=True)
                desired = torch.clamp(norm, 0, self.max_val)
                module.weight *= (desired / (norm + 1e-8))


def _apply_maxnorm_fc_heads(model: nn.Module) -> None:
    if hasattr(model, "fc_motor") and hasattr(model, "fc_lat"):
        model.fc_motor.apply(MaxNormConstraint(0.25))
        model.fc_lat.apply(MaxNormConstraint(0.25))
    else:
        model.fc.apply(MaxNormConstraint(0.25))


class EEGNetEncoder(nn.Module):
    """Blocos convolucionais EEGNet (Lawhern et al.); saída achatada ``(batch, feature_size)``."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 480,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.conv1 = nn.Conv2d(1, F1, (1, 80), padding=(0, 40), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        self.feature_size = self._infer_feature_size()

    def _infer_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            return self.forward(x).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x.view(x.size(0), -1)


class EEGNetModel(nn.Module):
    """
    EEGNet architecture.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    n_samples : int
        Number of time samples per epoch
    n_classes : int
        Number of output classes
    F1 : int
        Number of temporal filters
    D : int
        Depth multiplier for depthwise convolution
    F2 : int
        Number of pointwise filters
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 480,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5
    ):
        super(EEGNetModel, self).__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            F1=F1,
            D=D,
            F2=F2,
            dropout=dropout,
        )
        self.feature_size = self.encoder.feature_size
        self.fc = nn.Linear(self.feature_size, n_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class EEGNetSetup1HierModel(nn.Module):
    """
    Setup 1 (planning/07): backbone EEGNet partilhado + duas cabeças binárias
    (repouso vs. motora; esquerda vs. direita), alinhado a CSP/LDA e FBCSP v1.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 480,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            F1=F1,
            D=D,
            F2=F2,
            dropout=dropout,
        )
        fs = self.encoder.feature_size
        self.fc_motor = nn.Linear(fs, 2)
        self.fc_lat = nn.Linear(fs, 2)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x)
        return self.fc_motor(f), self.fc_lat(f)


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for EEGNet.
    
    Parameters
    ----------
    F1 : int
        Number of temporal filters
    D : int
        Depth multiplier
    F2 : int
        Number of pointwise filters
    dropout : float
        Dropout rate
    epochs : int
        Training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    filter_band : tuple or None
        Bandpass filter (low, high) in Hz
    sfreq : float
        Sampling frequency
    """
    
    def __init__(
        self,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        epochs: int = 150,
        lr: float = 0.001,
        batch_size: int = 16,
        use_abat: bool = False,
        filter_band: Optional[Tuple[float, float]] = (4, 40),
        sfreq: float = 160.0,
        patience: int = 20,
        val_split: float = 0.15,
        optimizer: str = 'adam',
        weight_decay: float = 1e-4,
        n_classes: Optional[int] = None,
    ):
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.use_abat = use_abat
        self.filter_band = filter_band
        self.sfreq = sfreq
        self.patience = patience
        self.val_split = val_split
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.n_classes = n_classes

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None
        
        # Setup filter
        self.filter_b = None
        self.filter_a = None
        if self.filter_band is not None:
            nyq = self.sfreq / 2.0
            low = self.filter_band[0] / nyq
            high = self.filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(5, [low, high], btype='band')
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Apply CAR and bandpass filter."""
        X = common_average_reference(X)
        if self.filter_b is not None:
            X_filtered = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_filtered[i] = signal.filtfilt(self.filter_b, self.filter_a, X[i], axis=1)
            return X_filtered
        return X
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_epochs, n_channels, n_samples)
        y : np.ndarray
            Labels
        """
        # Store classes
        self.classes_ = np.unique(y)
        
        # Preprocess
        X = self._preprocess(X)
        
        # Get dimensions
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = (
            int(self.n_classes) if self.n_classes is not None else len(self.classes_)
        )
        
        # Create model
        self.model = EEGNetModel(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            dropout=self.dropout
        ).to(self.device)
        
        # Train/val split for early stopping
        n_val = max(1, int(len(X) * self.val_split))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train_t = torch.LongTensor(y[train_idx]).to(self.device)
        X_val_t   = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val_t   = torch.LongTensor(y[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size, shuffle=True
        )

        criterion = nn.CrossEntropyLoss()
        
        if self.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = float('inf')
        best_weights = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                if self.use_abat:
                    batch_X.requires_grad = True
                    
                optimizer.zero_grad(set_to_none=True)
                out = self.model(batch_X)
                loss_clean = criterion(out, batch_y)
                
                if self.use_abat:
                    # Backward pass to get the gradient of the input image
                    loss_clean.backward(retain_graph=True)
                    grad_sign = torch.sign(batch_X.grad)
                    
                    # Generate Adversarial Example (FGSM)
                    epsilon = 0.03 # As found effective in the ABAT paper
                    X_adv = batch_X + epsilon * grad_sign
                    X_adv = X_adv.detach()
                    
                    # Secondary Forward pass on adversarial examples
                    out_adv = self.model(X_adv)
                    loss_adv = criterion(out_adv, batch_y)
                    
                    # Accumulate and optimize
                    loss_total = loss_clean + loss_adv
                    loss_total.backward()
                else:
                    loss_clean.backward()
                    
                optimizer.step()
                
                # Apply MaxNorm constraints (Lawhern et al. 2018)
                self.model.encoder.conv2.apply(MaxNormConstraint(1.0))
                _apply_maxnorm_fc_heads(self.model)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        # Restore best weights
        self.model.load_state_dict(best_weights)

        return self

    def continue_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> "EEGNetClassifier":
        """
        Continua o treino no mesmo ``nn.Module`` (p.ex. *fine-tuning* MI após movimento real).

        Exige ``fit`` prévio. Mantém dimensões de entrada e número de classes do modelo atual.
        """
        if self.model is None:
            raise RuntimeError("continue_training requires fit() first")

        X = self._preprocess(X)
        train_lr = self.lr if lr is None else lr
        train_patience = self.patience if patience is None else patience

        n_val = max(1, int(len(X) * self.val_split))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train_t = torch.LongTensor(y[train_idx]).to(self.device)
        X_val_t = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val_t = torch.LongTensor(y[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        criterion = nn.CrossEntropyLoss()
        if self.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(), lr=train_lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(), lr=train_lr, weight_decay=self.weight_decay
            )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                out = self.model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                self.model.encoder.conv2.apply(MaxNormConstraint(1.0))
                _apply_maxnorm_fc_heads(self.model)

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= train_patience:
                    break

        self.model.load_state_dict(best_weights)
        self.classes_ = np.unique(
            np.concatenate([self.classes_, np.unique(y)])
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X = self._preprocess(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._preprocess(X)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)
            return proba.cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class EEGNetSetup1Classifier(EEGNetClassifier):
    """
    EEGNet para **Setup 1** (planning/07): backbone partilhado + cabeça **motor** (repouso vs.
    motora) + cabeça **lateral** (esq. vs. dir.), com treino em duas fases via ``fit`` + ``continue_training``.
    """

    @staticmethod
    def _loss_setup1(
        om: torch.Tensor,
        ol: torch.Tensor,
        batch_y: torch.Tensor,
        crit: nn.Module,
    ) -> torch.Tensor:
        ym = (batch_y > 0).long()
        loss_m = crit(om, ym)
        mask = batch_y > 0
        if mask.any():
            loss_l = crit(ol[mask], batch_y[mask] - 1)
        else:
            loss_l = loss_m * 0.0
        return loss_m + loss_l

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetSetup1Classifier":
        self.classes_ = np.array([0, 1, 2])
        X = self._preprocess(X)
        y = np.asarray(y).astype(np.int64).ravel()
        n_channels, n_samples = X.shape[1], X.shape[2]
        self.model = EEGNetSetup1HierModel(
            n_channels=n_channels,
            n_samples=n_samples,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            dropout=self.dropout,
        ).to(self.device)

        n_val = max(1, int(len(X) * self.val_split))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train_t = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train_t = torch.LongTensor(y[train_idx]).to(self.device)
        X_val_t = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val_t = torch.LongTensor(y[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        crit = nn.CrossEntropyLoss()
        if self.optimizer.lower() == "adamw":
            opt = optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            opt = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                opt.zero_grad(set_to_none=True)
                om, ol = self.model(batch_X)
                loss = self._loss_setup1(om, ol, batch_y, crit)
                loss.backward()
                opt.step()
                self.model.encoder.conv2.apply(MaxNormConstraint(1.0))
                _apply_maxnorm_fc_heads(self.model)

            self.model.eval()
            with torch.no_grad():
                om, ol = self.model(X_val_t)
                vloss = self._loss_setup1(om, ol, y_val_t, crit).item()

            if vloss < best_val:
                best_val = vloss
                best_w = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        self.model.load_state_dict(best_w)
        return self

    def continue_training(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 80,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
    ) -> "EEGNetSetup1Classifier":
        if self.model is None:
            raise RuntimeError("continue_training requires fit() first")
        if not isinstance(self.model, EEGNetSetup1HierModel):
            raise TypeError("continue_training Setup1 espera EEGNetSetup1HierModel")

        X = self._preprocess(X)
        y = np.asarray(y).astype(np.int64).ravel()
        train_lr = self.lr if lr is None else lr
        train_patience = self.patience if patience is None else patience

        n_val = max(1, int(len(X) * self.val_split))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        X_train_t = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train_t = torch.LongTensor(y[train_idx]).to(self.device)
        X_val_t = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val_t = torch.LongTensor(y[val_idx]).to(self.device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
        )
        crit = nn.CrossEntropyLoss()
        if self.optimizer.lower() == "adamw":
            opt = optim.AdamW(
                self.model.parameters(), lr=train_lr, weight_decay=self.weight_decay
            )
        else:
            opt = optim.Adam(
                self.model.parameters(), lr=train_lr, weight_decay=self.weight_decay
            )

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                opt.zero_grad(set_to_none=True)
                om, ol = self.model(batch_X)
                loss = self._loss_setup1(om, ol, batch_y, crit)
                loss.backward()
                opt.step()
                self.model.encoder.conv2.apply(MaxNormConstraint(1.0))
                _apply_maxnorm_fc_heads(self.model)

            self.model.eval()
            with torch.no_grad():
                om, ol = self.model(X_val_t)
                vloss = self._loss_setup1(om, ol, y_val_t, crit).item()

            if vloss < best_val:
                best_val = vloss
                best_w = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= train_patience:
                    break

        self.model.load_state_dict(best_w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess(X)
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt)
            pm = om.argmax(dim=1)
            out = torch.zeros(len(pm), dtype=torch.long, device=self.device)
            rest = pm == 0
            out[rest] = 0
            mot = pm == 1
            if mot.any():
                out[mot] = 1 + ol[mot].argmax(dim=1)
            return out.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Heurística P(0), P(1), P(2) a partir das softmax das duas cabeças."""
        X = self._preprocess(X)
        self.model.eval()
        with torch.no_grad():
            xt = torch.FloatTensor(X).to(self.device)
            om, ol = self.model(xt)
            sm_m = torch.softmax(om, dim=1)
            sm_l = torch.softmax(ol, dim=1)
            p0 = sm_m[:, 0]
            p_motor = sm_m[:, 1]
            n = len(p0)
            proba = torch.zeros(n, 3, device=self.device)
            proba[:, 0] = p0
            proba[:, 1] = p_motor * sm_l[:, 0]
            proba[:, 2] = p_motor * sm_l[:, 1]
            proba /= proba.sum(dim=1, keepdim=True).clamp(min=1e-8)
            return proba.cpu().numpy()


def create_eegnet_setup1_v7_model(
    F1: int = 16,
    D: int = 4,
    F2: int = 64,
    dropout: float = 0.5,
    epochs: int = 150,
    lr: float = 0.001,
    batch_size: int = 16,
    patience: int = 20,
    optimizer: str = "adam",
    weight_decay: float = 1e-4,
    filter_band: Optional[Tuple[float, float]] = (4, 40),
    val_split: float = 0.15,
) -> EEGNetSetup1Classifier:
    """EEGNet v7 (F1/D/F2/dropout) no desenho Setup 1 (duas cabeças).

    ``filter_band`` e ``val_split`` passaram a ser parâmetros (antes fixos no corpo)
    para a correção B-6: sob o contrato de harmonização a filtragem tem dono único, o
    harness, e o modelo é construído com ``filter_band=None`` para não poder filtrar de
    novo. Os defaults preservam o comportamento anterior para os chamadores antigos.
    """
    return EEGNetSetup1Classifier(
        F1=F1,
        D=D,
        F2=F2,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_abat=False,
        patience=patience,
        optimizer=optimizer,
        weight_decay=weight_decay,
        filter_band=filter_band,
        val_split=val_split,
        sfreq=160.0,
        n_classes=None,
    )


def create_eegnet_model(
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    dropout: float = 0.5,
    epochs: int = 150,
    lr: float = 0.001,
    batch_size: int = 16,
    use_abat: bool = False,
    patience: int = 20,
    optimizer: str = 'adam',
    weight_decay: float = 1e-4,
    n_classes: Optional[int] = None,
) -> EEGNetClassifier:
    """
    Factory function to create EEGNet classifier.
    
    Parameters
    ----------
    F1 : int
        Temporal filters (paper: 8)
    D : int
        Depth multiplier (paper: 2)
    F2 : int
        Pointwise filters (paper: 16)
    dropout : float
        Dropout rate (paper: 0.5)
    epochs : int
        Training epochs
    lr : float
        Learning rate (v0.4: 0.001)
    batch_size : int
        Batch size (v0.4: 16)
    use_abat : bool
        Activate ABAT (Adversarial Training + Euclidean Alignment)
    patience : int
        Early stopping patience
    optimizer : str
        Optimizer to use ('adam' or 'adamw')
    weight_decay : float
        L2 penalty (default: 1e-4)
        
    Returns
    -------
    model : EEGNetClassifier
        Fresh model instance
    """
    return EEGNetClassifier(
        F1=F1,
        D=D,
        F2=F2,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_abat=use_abat,
        patience=patience,
        filter_band=(4, 40),
        sfreq=160.0,
        optimizer=optimizer,
        weight_decay=weight_decay,
        n_classes=n_classes,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from core.data_loader import load_subject
    from core.metrics import compute_metrics, format_metrics
    from sklearn.model_selection import train_test_split
    
    print("Testing EEGNet on Subject 1...")
    
    # Load data
    X, y = load_subject(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create and train model
    model = create_eegnet_model(epochs=100)  # Fewer epochs for testing
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    print(f"\nResults: {format_metrics(metrics)}")
