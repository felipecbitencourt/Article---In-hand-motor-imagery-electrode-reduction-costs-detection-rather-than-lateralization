"""
Riemannian Geometry-based Classification with DCCA

Uses covariance matrices on the Riemannian manifold with MDM classifier.
Optionally includes Detrended Cross-Correlation Analysis (DCCA) for improved features.

Reference: Combining detrended cross-correlation analysis with Riemannian 
           geometry-based classification for improved MI-BCI

Paper Parameters:
- Channels: 22 (motor cortex subset)
- Bandpass: 8-30 Hz
- Covariance estimation: Sample covariance or DCCA-based
- Classifier: MDM (Minimum Distance to Mean)
- Validation: Leave-One-Run-Out (LORO-CV)

Setup 1 (planning/07): ``HierarchicalRiemannianSetup1V1`` — duas cabeças (motor + lateral),
espaço tangente estimado na Fase 1 (real); na Fase 2 (MI) **referência tangente congelada** e
só **logística** reajustada; covariâncias Ledoit–Wolf reajustadas em cada fase nos dados respetivos.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, ClassifierMixin
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Optional, List

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from core.preprocessing import common_average_reference


# Motor cortex channels (22 channels as in the paper)
MOTOR_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'C3', 'Cz', 'C4',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P7', 'P3', 'Pz', 'P4', 'P8'
]

# PhysioNet 64 channel indices for motor cortex (approximate)
# Based on 10-10 system positioning
#
# LEGADO — NAO USAR EM BENCHMARK DE DENSIDADE (correcao B-2).
# Sao indices POSICIONAIS no arranjo de 64 canais. Quando o harness ja entregou uma
# montagem reduzida, esses indices apontam para outros eletrodos, e `_select_channels`
# ainda descarta em silencio os que caem fora (`i < X.shape[1]`). O efeito medido: com
# a montagem de 8/21/64 canais o modelo via 7/18/23 canais, ou seja, recortava por
# cima do eixo que o benchmark mede. Selecao de canais e propriedade uniforme do
# contrato e pertence ao harness; nenhum modelo pode recortar de novo.
# Mantido apenas para reproduzir resultados antigos de 64 canais.
PHYSIONET_MOTOR_INDICES = [
    0, 1, 2, 3, 4, 5, 6,  # Frontal
    8, 9, 10, 11,  # FC
    13, 14, 15,  # Central (C3, Cz, C4)
    17, 18, 19, 20,  # CP
    22, 23, 24, 25, 26  # Parietal
]


class RiemannianClassifier(BaseEstimator, ClassifierMixin):
    """
    Riemannian geometry-based classifier for EEG.
    
    Uses covariance matrices projected onto the Riemannian manifold
    and MDM (Minimum Distance to Mean) or Tangent Space + LR classifier.
    
    Parameters
    ----------
    method : str
        'mdm' for Minimum Distance to Mean
        'tangent' for Tangent Space + Logistic Regression
    metric : str
        Riemannian metric ('riemann', 'logeuclid', 'euclid')
    filter_band : tuple
        Bandpass filter (low, high) in Hz, or None
    sfreq : float
        Sampling frequency
    channel_indices : list or None
        Indices of channels to use, None = all
    """
    
    def __init__(
        self,
        method: str = 'mdm',
        metric: str = 'riemann',
        filter_band: Optional[tuple] = (8, 30),
        sfreq: float = 160.0,
        channel_indices: Optional[List[int]] = None
    ):
        self.method = method
        self.metric = metric
        self.filter_band = filter_band
        self.sfreq = sfreq
        self.channel_indices = channel_indices
        
        self.cov_estimator = Covariances(estimator='lwf')  # Ledoit-Wolf
        self.classifier = None
        self.filter_b = None
        self.filter_a = None
        
        self._setup_filter()
        self._setup_classifier()
    
    def _setup_filter(self):
        """Setup bandpass filter."""
        if self.filter_band is not None:
            nyq = self.sfreq / 2.0
            low = self.filter_band[0] / nyq
            high = self.filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(5, [low, high], btype='band')
    
    def _setup_classifier(self):
        """Setup the classifier pipeline."""
        if self.method == 'mdm':
            self.classifier = MDM(metric=self.metric)
        elif self.method == 'tangent':
            self.classifier = Pipeline([
                ('ts', TangentSpace(metric=self.metric)),
                ('lr', LogisticRegression(max_iter=1000))
            ])
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing: CAR, channel selection and filtering."""
        # Step 1: Common Average Reference (remove shared noise before covariance)
        X = common_average_reference(X)

        # Step 2: Channel selection
        if self.channel_indices is not None:
            valid_indices = [i for i in self.channel_indices if i < X.shape[1]]
            if len(valid_indices) > 0:
                X = X[:, valid_indices, :]

        # Step 3: Bandpass filter
        if self.filter_b is not None:
            X_filtered = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_filtered[i] = signal.filtfilt(self.filter_b, self.filter_a, X[i], axis=1)
            X = X_filtered

        return X
    
    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrices for all epochs."""
        return self.cov_estimator.fit_transform(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_epochs, n_channels, n_samples)
        y : np.ndarray
            Labels
        """
        # Preprocess
        X = self._preprocess(X)
        
        # Compute covariance matrices
        covs = self._compute_covariances(X)
        
        # Fit classifier
        self.classifier.fit(covs, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        X = self._preprocess(X)
        covs = self._compute_covariances(X)
        return self.classifier.predict(covs)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = self._preprocess(X)
        covs = self._compute_covariances(X)
        
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(covs)
        else:
            # MDM doesn't have predict_proba by default
            # Return distances as pseudo-probabilities
            predictions = self.classifier.predict(covs)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 1
            proba[predictions == 1, 1] = 1
            return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class HierarchicalRiemannianSetup1V1:
    """
    Setup 1 hierárquico: repouso (0) vs. MI/motora (1∪2), depois esquerda (1) vs. direita (2).

    Usa **Tangent Space + regressão logística** em cada cabeça (``method='tangent'`` recomendado
    para Fase 2 parcial). O ``TangentSpace`` é ajustado só na **Fase 1**; na **Fase 2** apenas
    ``LogisticRegression`` é reajustado sobre ``ts.transform(covs)``.

    Quando ``preprocessed_externally=True``, assume-se entrada já com CAR + passa-banda (p.ex.
    ``preprocess_epochs`` 4–40 Hz); aplica-se só a seleção de canais motores.
    """

    def __init__(
        self,
        metric: str = "riemann",
        sfreq: float = 160.0,
        channel_indices: Optional[List[int]] = None,
        preprocessed_externally: bool = True,
        filter_band: Optional[tuple] = None,
        random_state: Optional[int] = None,
        lr_max_iter: int = 2000,
    ):
        self.metric = metric
        self.sfreq = sfreq
        self.channel_indices = channel_indices
        self.preprocessed_externally = preprocessed_externally
        self.filter_band = filter_band
        self.random_state = random_state
        self.lr_max_iter = lr_max_iter

        self.cov_motor = Covariances(estimator="lwf")
        self.cov_lat = Covariances(estimator="lwf")
        self.ts_motor = TangentSpace(metric=metric)
        self.ts_lat = TangentSpace(metric=metric)
        self.lr_motor = LogisticRegression(
            max_iter=lr_max_iter, random_state=random_state
        )
        self.lr_lat = LogisticRegression(
            max_iter=lr_max_iter, random_state=random_state
        )
        self._phase1_fitted = False
        self.filter_b = None
        self.filter_a = None
        if not preprocessed_externally and filter_band is not None:
            nyq = sfreq / 2.0
            lo = filter_band[0] / nyq
            hi = filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(5, [lo, hi], btype="band")

    def _select_channels(self, X: np.ndarray) -> np.ndarray:
        if self.channel_indices is None:
            return X
        valid = [i for i in self.channel_indices if i < X.shape[1]]
        if not valid:
            return X
        return X[:, valid, :]

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        if self.preprocessed_externally:
            return self._select_channels(X.astype(np.float64))
        X = common_average_reference(X.astype(np.float64))
        X = self._select_channels(X)
        if self.filter_b is not None:
            out = np.zeros_like(X)
            for i in range(X.shape[0]):
                out[i] = signal.filtfilt(self.filter_b, self.filter_a, X[i], axis=1)
            X = out
        return X

    def fit_phase1_real(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalRiemannianSetup1V1":
        y = np.asarray(y).astype(np.int64).ravel()
        Xp = self._preprocess(X)

        y_m = (y > 0).astype(np.int64)
        self.cov_motor.fit(Xp)
        cm = self.cov_motor.transform(Xp)
        self.ts_motor.fit(cm, y_m)
        self.lr_motor.fit(self.ts_motor.transform(cm), y_m)

        mask = y > 0
        if int(np.sum(mask)) < 4:
            raise ValueError(
                f"Fase 1: poucas épocas motoras para cabeça lateral (N={int(np.sum(mask))})."
            )
        Xl = Xp[mask]
        yl = (y[mask] - 1).astype(np.int64)
        self.cov_lat.fit(Xl)
        cl = self.cov_lat.transform(Xl)
        self.ts_lat.fit(cl, yl)
        self.lr_lat.fit(self.ts_lat.transform(cl), yl)
        self._phase1_fitted = True
        return self

    def fit_phase2_mi(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalRiemannianSetup1V1":
        if not self._phase1_fitted:
            raise RuntimeError("Chame fit_phase1_real antes de fit_phase2_mi.")

        y = np.asarray(y).astype(np.int64).ravel()
        Xp = self._preprocess(X)

        y_m = (y > 0).astype(np.int64)
        self.cov_motor.fit(Xp)
        cm = self.cov_motor.transform(Xp)
        self.lr_motor.fit(self.ts_motor.transform(cm), y_m)

        mask = y > 0
        if int(np.sum(mask)) < 2:
            raise ValueError("Fase 2: sem épocas MI para cabeça lateral.")
        Xl = Xp[mask]
        yl = (y[mask] - 1).astype(np.int64)
        self.cov_lat.fit(Xl)
        cl = self.cov_lat.transform(Xl)
        self.lr_lat.fit(self.ts_lat.transform(cl), yl)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xp = self._preprocess(X)
        cm = self.cov_motor.transform(Xp)
        pm = self.lr_motor.predict(self.ts_motor.transform(cm)).astype(np.int64)

        out = np.zeros(len(X), dtype=np.int64)
        rest = pm == 0
        out[rest] = 0
        idx = np.where(~rest)[0]
        if idx.size > 0:
            cl = self.cov_lat.transform(Xp[idx])
            pl = self.lr_lat.predict(self.ts_lat.transform(cl)).astype(np.int64)
            out[idx] = 1 + pl
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior de 3 classes a partir das duas regressões logísticas (correção B-4).

        Mesma composição em cascata usada por ``csp_lda`` e ``fbcsp_svm``, de modo que
        as três saídas macias sejam comparáveis entre si:
        ``P(0)=P(repouso)``, ``P(1)=P(motor)·P(esq|motor)``, ``P(2)=P(motor)·P(dir|motor)``.

        A cabeça lateral é avaliada em TODAS as épocas, e não só nas classificadas como
        motoras: a cascata dura zeraria a probabilidade das classes laterais sempre que
        o Nível 1 dissesse repouso, e AUC e log-loss ficariam degeneradas justamente nas
        épocas em que o Nível 1 erra, que são as que interessam medir.
        """
        Xp = self._preprocess(X)
        cm = self.cov_motor.transform(Xp)
        p_rm = self.lr_motor.predict_proba(self.ts_motor.transform(cm))
        i0 = int(np.where(self.lr_motor.classes_ == 0)[0][0])
        i1 = int(np.where(self.lr_motor.classes_ == 1)[0][0])

        cl = self.cov_lat.transform(Xp)
        p_lat = self.lr_lat.predict_proba(self.ts_lat.transform(cl))
        j0 = int(np.where(self.lr_lat.classes_ == 0)[0][0])
        j1 = int(np.where(self.lr_lat.classes_ == 1)[0][0])

        proba = np.zeros((len(X), 3), dtype=np.float64)
        proba[:, 0] = p_rm[:, i0]
        proba[:, 1] = p_rm[:, i1] * p_lat[:, j0]
        proba[:, 2] = p_rm[:, i1] * p_lat[:, j1]
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return proba / s

    # Alias: o harness admite modelo na tabela primaria so com saida macia.
    predict_proba_three_class = predict_proba


def create_riemannian_setup1_v1_model(
    metric: str = "riemann",
    use_channel_subset: bool = False,
    random_state: Optional[int] = None,
    preprocessed_externally: bool = True,
) -> HierarchicalRiemannianSetup1V1:
    """Fábrica para o decoder Riemanniano hierárquico Setup 1 (v1).

    ``use_channel_subset`` tem default ``False`` desde a correção B-2: quem seleciona
    canais é o harness. Ver a nota em ``PHYSIONET_MOTOR_INDICES``.
    """
    ch = PHYSIONET_MOTOR_INDICES if use_channel_subset else None
    return HierarchicalRiemannianSetup1V1(
        metric=metric,
        channel_indices=ch,
        preprocessed_externally=preprocessed_externally,
        random_state=random_state,
    )


def create_riemannian_model(
    method: str = 'tangent',
    metric: str = 'riemann',
    use_channel_subset: bool = True
) -> RiemannianClassifier:
    """
    Factory function to create Riemannian classifier.
    
    Parameters
    ----------
    method : str
        'mdm' or 'tangent'
    metric : str
        'riemann', 'logeuclid', or 'euclid'
    use_channel_subset : bool
        If True, use 22-channel motor cortex subset
        
    Returns
    -------
    model : RiemannianClassifier
        Fresh model instance
    """
    channel_indices = PHYSIONET_MOTOR_INDICES if use_channel_subset else None
    
    return RiemannianClassifier(
        method=method,
        metric=metric,
        filter_band=(8, 30),  # Paper uses 8-30 Hz
        sfreq=160.0,
        channel_indices=channel_indices
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from core.data_loader import load_subject
    from core.metrics import compute_metrics, format_metrics
    from sklearn.model_selection import train_test_split
    
    print("Testing Riemannian classifier on Subject 1...")
    
    # Load data
    X, y = load_subject(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Test MDM classifier
    print("\n--- MDM Classifier ---")
    model = create_riemannian_model(method='mdm')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    print(f"Results: {format_metrics(metrics)}")
    
    # Test Tangent Space + LR
    print("\n--- Tangent Space + LR ---")
    model_ts = create_riemannian_model(method='tangent')
    model_ts.fit(X_train, y_train)
    y_pred_ts = model_ts.predict(X_test)
    
    metrics_ts = compute_metrics(y_test, y_pred_ts)
    print(f"Results: {format_metrics(metrics_ts)}")
