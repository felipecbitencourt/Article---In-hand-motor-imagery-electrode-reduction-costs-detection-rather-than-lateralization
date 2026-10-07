"""
Filter Bank Common Spatial Patterns (FBCSP) + SVM

Extension of CSP that decomposes the signal into multiple frequency bands,
applies CSP to each band, and uses feature selection (MIBIF) before SVM classification.

Reference: Decoding the EEG patterns induced by sequential finger movement for BCIs

Paper Parameters:
- Filter Bank: 4-30 Hz (7 sub-bands of 4 Hz each)
- CSP: 2 components per band = 14 features total
- Feature Selection: Mutual Information (MIBIF), select 10 features
- Classifier: SVM with C=1
- Validation: 10-fold cross-validation

Setup 1 (planning/07): ``HierarchicalFBCSPSetup1V1`` — duas pilhas FBCSP+SVM hierárquicas
(real → MI, FBCSP congelado na fase 2).
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from mne.decoding import CSP
from typing import List, Tuple, Optional

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from core.preprocessing import common_average_reference


def _posto_efetivo(X: np.ndarray, tol: float = 1e-9) -> int:
    """Posto numerico da covariancia media entre canais (correcao D-4).

    Sob CAR o posto e ``n_canais - 1``: a referencia media comum zera a direcao de
    modo comum, e ela continua nula depois de qualquer filtro temporal aplicado
    igualmente a todos os canais. Medido em vez de assumido porque o pipeline pode
    entregar dados sem CAR.
    """
    amostra = X[:: max(1, len(X) // 64)]
    C = np.mean([np.cov(x) for x in amostra], axis=0)
    w = np.linalg.eigvalsh(C)
    return int(np.sum(w > tol * max(w.max(), np.finfo(float).tiny)))


class FilterBank:
    """
    Filter bank for decomposing EEG into multiple frequency bands.
    
    Parameters
    ----------
    freq_bands : list of tuple
        List of (low, high) frequency pairs
    sfreq : float
        Sampling frequency
    order : int
        Filter order
    """
    
    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        sfreq: float = 160.0,
        order: int = 5
    ):
        if freq_bands is None:
            # Default: 6 sub-bands covering mu and beta motor bands (8-30 Hz)
            self.freq_bands = [
                (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)
            ]
        else:
            self.freq_bands = freq_bands
        
        self.sfreq = sfreq
        self.order = order
        self.filters = self._design_filters()
    
    def _design_filters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Design bandpass filters for each frequency band."""
        filters = []
        nyq = self.sfreq / 2.0
        
        for low, high in self.freq_bands:
            low_norm = low / nyq
            high_norm = high / nyq
            
            # Ensure frequencies are valid
            low_norm = max(0.001, min(low_norm, 0.999))
            high_norm = max(low_norm + 0.001, min(high_norm, 0.999))
            
            b, a = signal.butter(self.order, [low_norm, high_norm], btype='band')
            filters.append((b, a))
        
        return filters
    
    def filter(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Apply filter bank to data.
        
        Parameters
        ----------
        X : np.ndarray
            EEG data, shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        X_filtered : list of np.ndarray
            List of filtered data for each frequency band
        """
        X_filtered = []
        
        for b, a in self.filters:
            X_band = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_band[i] = signal.filtfilt(b, a, X[i], axis=1)
            X_filtered.append(X_band)
        
        return X_filtered


class FBCSP:
    """
    Filter Bank CSP feature extractor.
    
    Applies CSP to each frequency band and concatenates features.
    
    Parameters
    ----------
    freq_bands : list of tuple
        Frequency bands for filter bank
    n_components : int
        Number of CSP components per band
    sfreq : float
        Sampling frequency
    """
    
    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        n_components: int = 2,
        sfreq: float = 160.0
    ):
        self.freq_bands = freq_bands
        self.n_components = n_components
        self.n_components_efetivo = n_components   # ajustado ao posto no fit (D-4)
        self.sfreq = sfreq

        self.filter_bank = FilterBank(freq_bands=freq_bands, sfreq=sfreq)
        self.csp_list = []
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP for each frequency band.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_epochs, n_channels, n_samples)
        y : np.ndarray
            Labels
        """
        # Apply CAR before filter bank
        X = common_average_reference(X)
        
        # Filter into bands
        X_bands = self.filter_bank.filter(X)
        
        # Correcao D-4: limita os filtros por sub-banda ao posto disponivel.
        #
        # Sob CAR o posto e n_canais - 1, e pedir mais componentes que isso faz o CSP
        # devolver direcoes de variancia nula; com log=True a feature vira log do erro
        # de arredondamento. Aqui o efeito era pior que no csp_lda porque o
        # SelectKBest ESCOLHE essas features: medido em 2 e 3 canais, 6 das 12
        # selecionadas eram as degeneradas, porque correlacionam 0,87-0,93 com a
        # log-potencia da epoca. O tipo de feature mudava com a densidade, que e
        # exatamente o eixo sob teste.
        self.n_components_efetivo = max(1, min(self.n_components, _posto_efetivo(X)))

        # Fit CSP for each band
        self.csp_list = []
        for X_band in X_bands:
            csp = CSP(
                n_components=self.n_components_efetivo,
                reg='ledoit_wolf',
                log=True,
                norm_trace=True  # Better inter-subject generalization
            )
            csp.fit(X_band, y)
            self.csp_list.append(csp)

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted CSP models.
        
        Parameters
        ----------
        X : np.ndarray
            Data, shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        features : np.ndarray
            CSP features, shape (n_epochs, n_bands * n_components)
        """
        # Apply CAR before filter bank
        X = common_average_reference(X)
        
        X_bands = self.filter_bank.filter(X)
        
        features_list = []
        for X_band, csp in zip(X_bands, self.csp_list):
            features = csp.transform(X_band)
            features_list.append(features)
        
        return np.hstack(features_list)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


class FBCSPSVMClassifier:
    """
    Complete FBCSP + SVM pipeline.
    
    Parameters
    ----------
    freq_bands : list of tuple, optional
        Frequency bands. Default: 9 bands from 4-40 Hz
    n_csp_components : int
        CSP components per band
    n_features : int or None
        Number of features to select. None = use all
    sfreq : float
        Sampling frequency
    svm_C : float
        SVM regularization parameter
    svm_kernel : str
        SVM kernel type
    """
    
    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        n_csp_components: int = 2,
        n_features: Optional[int] = None,
        sfreq: float = 160.0,
        svm_C: float = 1.0,
        svm_kernel: str = 'rbf'
    ):
        self.freq_bands = freq_bands
        self.n_csp_components = n_csp_components
        self.n_features = n_features
        self.sfreq = sfreq
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel
        
        self.fbcsp = FBCSP(
            freq_bands=freq_bands,
            n_components=n_csp_components,
            sfreq=sfreq
        )
        self.scaler = StandardScaler()
        self.selector = None
        self.svm = SVC(C=svm_C, kernel=svm_kernel, probability=True)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the complete pipeline.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_epochs, n_channels, n_samples)
        y : np.ndarray
            Labels
        """
        # Extract FBCSP features
        features = self.fbcsp.fit_transform(X, y)
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Feature selection using mutual information
        if self.n_features is not None and self.n_features < features.shape[1]:
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=self.n_features
            )
            features = self.selector.fit_transform(features, y)
        
        # Train SVM
        self.svm.fit(features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        features = self.fbcsp.transform(X)
        features = self.scaler.transform(features)
        
        if self.selector is not None:
            features = self.selector.transform(features)
        
        return self.svm.predict(features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        features = self.fbcsp.transform(X)
        features = self.scaler.transform(features)
        
        if self.selector is not None:
            features = self.selector.transform(features)
        
        return self.svm.predict_proba(features)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


DEFAULT_SETUP1_FREQ_BANDS: List[Tuple[float, float]] = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
]


class HierarchicalFBCSPSetup1V1:
    """
    FBCSP + SVM hierárquico para o Setup 1 (rótulos 0=repouso, 1=esq., 2=dir.).

    Duas pilhas binárias (motor: repouso vs. T1∪T2; lateral: esq. vs. dir.), cada uma com
    **filter bank + CSP por banda + StandardScaler + SVM** (sem MIBIF por defeito, alinhado à v0.2).

    **Fase 1:** ajusta FBCSP, escalador, seletor (opcional) e SVM em dados **reais**.
    **Fase 2:** mantém os **FBCSP** (filtros por banda) e **reajusta** escalador, seletor e SVM em **MI**.
    """

    def __init__(
        self,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
        n_csp_components: int = 2,
        n_features: Optional[int] = None,
        sfreq: float = 160.0,
        svm_C: float = 1.0,
        svm_kernel: str = "linear",
    ):
        self.freq_bands = freq_bands if freq_bands is not None else list(DEFAULT_SETUP1_FREQ_BANDS)
        self.n_csp_components = n_csp_components
        self.n_features = n_features
        self.sfreq = sfreq
        self.svm_C = svm_C
        self.svm_kernel = svm_kernel

        self.fbcsp_motor = FBCSP(
            freq_bands=self.freq_bands,
            n_components=n_csp_components,
            sfreq=sfreq,
        )
        self.scaler_motor = StandardScaler()
        self.selector_motor: Optional[SelectKBest] = None

        self.fbcsp_lat = FBCSP(
            freq_bands=self.freq_bands,
            n_components=n_csp_components,
            sfreq=sfreq,
        )
        self.scaler_lat = StandardScaler()
        self.selector_lat: Optional[SelectKBest] = None
        self._phase1_fitted = False

        if svm_kernel == "liblinear":
            from sklearn.svm import LinearSVC
            self.svm_motor = LinearSVC(C=svm_C, max_iter=2000, random_state=42)
            self.svm_lat = LinearSVC(C=svm_C, max_iter=2000, random_state=42)
        else:
            self.svm_motor = SVC(C=svm_C, kernel=svm_kernel, probability=True)
            self.svm_lat = SVC(C=svm_C, kernel=svm_kernel, probability=True)

    def _apply_selector(
        self,
        selector: Optional[SelectKBest],
        X: np.ndarray,
        fit: bool,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if selector is None:
            return X
        if fit:
            assert y is not None
            return selector.fit_transform(X, y)
        return selector.transform(X)

    def _fit_svm_head(
        self,
        fbcsp: FBCSP,
        scaler: StandardScaler,
        svm: SVC,
        selector_store: str,
        X: np.ndarray,
        y_bin: np.ndarray,
        fit_fbcsp: bool,
    ) -> None:
        if fit_fbcsp:
            feat = fbcsp.fit_transform(X, y_bin)
        else:
            feat = fbcsp.transform(X)
        feat = scaler.fit_transform(feat)
        sel = None
        if self.n_features is not None and self.n_features < feat.shape[1]:
            sel = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
            feat = sel.fit_transform(feat, y_bin)
        svm.fit(feat, y_bin)
        if selector_store == "motor":
            self.selector_motor = sel
        else:
            self.selector_lat = sel

    def _predict_head(
        self,
        fbcsp: FBCSP,
        scaler: StandardScaler,
        selector: Optional[SelectKBest],
        svm: SVC,
        X: np.ndarray,
    ) -> np.ndarray:
        feat = fbcsp.transform(X)
        feat = scaler.transform(feat)
        feat = self._apply_selector(selector, feat, fit=False)
        return svm.predict(feat)

    def _proba_head(
        self,
        fbcsp: FBCSP,
        scaler: StandardScaler,
        selector: Optional[SelectKBest],
        svm,
        X: np.ndarray,
    ) -> np.ndarray:
        feat = fbcsp.transform(X)
        feat = scaler.transform(feat)
        feat = self._apply_selector(selector, feat, fit=False)
        if hasattr(svm, "predict_proba"):
            return svm.predict_proba(feat)
        else:
            dec = svm.decision_function(feat)
            if dec.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-dec))
                return np.vstack([1 - p, p]).T
            else:
                exp_dec = np.exp(dec)
                return exp_dec / np.sum(exp_dec, axis=1, keepdims=True)

    def fit_phase1_real(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalFBCSPSetup1V1":
        y = np.asarray(y).astype(np.int64).ravel()
        y_motor = (y > 0).astype(np.int64)
        self._fit_svm_head(
            self.fbcsp_motor,
            self.scaler_motor,
            self.svm_motor,
            "motor",
            X,
            y_motor,
            fit_fbcsp=True,
        )

        mask = y > 0
        n_m = int(np.sum(mask))
        need = max(4, 2 * self.n_csp_components * len(self.freq_bands))
        if n_m < need:
            raise ValueError(
                f"Fase 1: poucas épocas motoras para FBCSP lateral (N={n_m}, mín. sugerido {need})."
            )
        X_lat = X[mask]
        y_lat = (y[mask] - 1).astype(np.int64)
        self._fit_svm_head(
            self.fbcsp_lat,
            self.scaler_lat,
            self.svm_lat,
            "lat",
            X_lat,
            y_lat,
            fit_fbcsp=True,
        )
        self._phase1_fitted = True
        return self

    def fit_phase2_mi(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalFBCSPSetup1V1":
        if not self._phase1_fitted:
            raise RuntimeError("Chame fit_phase1_real antes de fit_phase2_mi.")

        y = np.asarray(y).astype(np.int64).ravel()
        y_motor = (y > 0).astype(np.int64)
        self._fit_svm_head(
            self.fbcsp_motor,
            self.scaler_motor,
            self.svm_motor,
            "motor",
            X,
            y_motor,
            fit_fbcsp=False,
        )

        mask = y > 0
        if int(np.sum(mask)) < 2:
            raise ValueError("Fase 2: sem épocas MI (y>0) para cabeça lateral.")
        X_lat = X[mask]
        y_lat = (y[mask] - 1).astype(np.int64)
        self._fit_svm_head(
            self.fbcsp_lat,
            self.scaler_lat,
            self.svm_lat,
            "lat",
            X_lat,
            y_lat,
            fit_fbcsp=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        motor = self._predict_head(
            self.fbcsp_motor,
            self.scaler_motor,
            self.selector_motor,
            self.svm_motor,
            X,
        ).astype(np.int64)

        out = np.zeros(len(X), dtype=np.int64)
        rest = motor == 0
        out[rest] = 0
        mot_idx = np.where(~rest)[0]
        if mot_idx.size > 0:
            lat = self._predict_head(
                self.fbcsp_lat,
                self.scaler_lat,
                self.selector_lat,
                self.svm_lat,
                X[mot_idx],
            ).astype(np.int64)
            out[mot_idx] = 1 + lat
        return out

    def predict_proba_three_class(self, X: np.ndarray) -> np.ndarray:
        p_rm = self._proba_head(
            self.fbcsp_motor,
            self.scaler_motor,
            self.selector_motor,
            self.svm_motor,
            X,
        )
        i0 = int(np.where(self.svm_motor.classes_ == 0)[0][0])
        i1 = int(np.where(self.svm_motor.classes_ == 1)[0][0])
        p_rest = p_rm[:, i0]
        p_motor = p_rm[:, i1]

        p_lat = self._proba_head(
            self.fbcsp_lat,
            self.scaler_lat,
            self.selector_lat,
            self.svm_lat,
            X,
        )
        j0 = int(np.where(self.svm_lat.classes_ == 0)[0][0])
        j1 = int(np.where(self.svm_lat.classes_ == 1)[0][0])

        n = len(X)
        proba = np.zeros((n, 3), dtype=np.float64)
        proba[:, 0] = p_rest
        proba[:, 1] = p_motor * p_lat[:, j0]
        proba[:, 2] = p_motor * p_lat[:, j1]
        s = proba.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        proba /= s
        return proba

    # Correcao B-4: ver a nota equivalente em csp_lda.py. Depende de `svm_kernel`
    # diferente de "liblinear" — o LinearSVC nao expoe predict_proba, e por isso o
    # harness recusa a admissao desse ramo na tabela primaria.
    predict_proba = predict_proba_three_class

    @property
    def n_components_efetivo(self) -> int:
        """Filtros CSP por sub-banda efetivamente usados (correcao D-4).

        Exposto para que o harness grave o valor no CSV: em montagens pequenas ele e
        menor que o declarado, e essa diferenca precisa ser visivel na tabela em vez
        de ficar implicita no codigo.
        """
        return int(getattr(self.fbcsp_motor, "n_components_efetivo", self.n_csp_components))


def create_fbcsp_svm_setup1_v1_model(
    n_csp_components: int = 2,
    n_features: Optional[int] = None,
    svm_C: float = 1.0,
    svm_kernel: str = "linear",
    freq_bands: Optional[List[Tuple[float, float]]] = None,
) -> HierarchicalFBCSPSetup1V1:
    """Fábrica para FBCSP+SVM hierárquico Setup 1 (v1)."""
    return HierarchicalFBCSPSetup1V1(
        freq_bands=freq_bands,
        n_csp_components=n_csp_components,
        n_features=n_features,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
    )


def create_fbcsp_svm_model(
    n_csp_components: int = 2,
    n_features: Optional[int] = None,  # v0.2: use all features (no premature selection)
    svm_C: float = 1.0,
    svm_kernel: str = 'linear'  # Linear kernel: more stable for inter-subject
) -> FBCSPSVMClassifier:
    """
    Factory function to create FBCSP+SVM model.
    
    Parameters
    ----------
    n_csp_components : int
        CSP components per frequency band
    n_features : int or None
        Number of features to select (None = all)
    svm_C : float
        SVM regularization
        
    Returns
    -------
    model : FBCSPSVMClassifier
        Fresh model instance
    """
    return FBCSPSVMClassifier(
        n_csp_components=n_csp_components,
        n_features=n_features,
        svm_C=svm_C,
        svm_kernel=svm_kernel
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from core.data_loader import load_subject
    from core.metrics import compute_metrics, format_metrics
    from sklearn.model_selection import train_test_split
    
    print("Testing FBCSP+SVM on Subject 1...")
    
    # Load data (no preprocessing needed - FBCSP does its own filtering)
    X, y = load_subject(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train and evaluate
    model = create_fbcsp_svm_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    print(f"\nResults: {format_metrics(metrics)}")
    print(f"\nTotal features: {len(model.fbcsp.freq_bands) * model.fbcsp.n_components}")
    print(f"Selected features: {model.n_features}")
