"""
CSP + LDA Baseline Model

Common Spatial Patterns (CSP) for feature extraction + Linear Discriminant Analysis.
This is the traditional baseline for motor imagery classification (~70% accuracy).

Setup 1 (planning/07): ``HierarchicalCSPLDASetup1V1`` — duas pilhas binárias
(repouso vs. motora; esquerda vs. direita) e treino em duas fases (real → MI).
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from typing import Optional


class CSPLDAClassifier:
    """
    CSP + LDA pipeline for EEG classification.
    
    Parameters
    ----------
    n_components : int
        Number of CSP components (spatial filters)
    reg : str or float
        Regularization for CSP. 'ledoit_wolf' recommended for small samples.
    """
    
    def __init__(self, n_components: int = 6, reg: str = 'ledoit_wolf'):
        self.n_components = n_components
        self.reg = reg
        self.csp = None
        self.lda = None
        self.pipeline = None
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the CSP + LDA pipeline."""
        self.csp = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=True,  # Log-variance features
            norm_trace=True  # Normalize trace for inter-subject generalization
        )
        self.lda = LinearDiscriminantAnalysis()
        
        self.pipeline = Pipeline([
            ('csp', self.csp),
            ('lda', self.lda)
        ])
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model.
        
        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_epochs, n_channels, n_samples)
        y : np.ndarray
            Labels, shape (n_epochs,)
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Parameters
        ----------
        X : np.ndarray
            Test data, shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Test data, shape (n_epochs, n_channels, n_samples)
            
        Returns
        -------
        proba : np.ndarray
            Class probabilities, shape (n_epochs, n_classes)
        """
        return self.pipeline.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return self.pipeline.score(X, y)
    
    def get_csp_patterns(self) -> np.ndarray:
        """Get CSP spatial patterns (for visualization)."""
        if self.csp is not None:
            return self.csp.patterns_
        return None
    
    def get_csp_filters(self) -> np.ndarray:
        """Get CSP spatial filters."""
        if self.csp is not None:
            return self.csp.filters_
        return None


class HierarchicalCSPLDASetup1V1:
    """
    CSP + LDA hierárquico para o Setup 1 (3 classes: repouso, MI esq., MI dir.).

    - **Nível 1:** CSP+LDA binário — repouso (rótulo 0) vs. tarefa motora/MI (rótulos 1 ou 2).
    - **Nível 2:** CSP+LDA binário — mão esquerda (1) vs. mão direita (2), treinado só em épocas
      com ``y > 0``.

    **Fase 1 (movimento real):** estima filtros espaciais CSP e LDA em dados R03/R07/R11.
    **Fase 2 (MI):** mantém os CSP da Fase 1 e **reajusta apenas os LDA** em dados R04/R08/R12,
    analogamente ao *fine-tuning* linear do decisor após pré-treino espacial em execução real.

    Os dados de entrada devem estar já pré-processados (p.ex. CAR + passa-banda), na mesma ordem
    que o script ``run_setup1_hier_csp_lda.py``.
    """

    def __init__(self, n_components: int = 6, reg: str = "ledoit_wolf", lda_solver: str = "svd"):
        self.n_components = n_components
        self.reg = reg
        self.lda_solver = lda_solver
        self.n_components_efetivo = n_components   # ajustado ao posto no primeiro fit
        self._monta_csp(n_components)
        self.lda_motor = LinearDiscriminantAnalysis(solver=lda_solver)
        self.lda_lateral = LinearDiscriminantAnalysis(solver=lda_solver)
        self._phase1_fitted = False

    def _monta_csp(self, k: int) -> None:
        self.csp_motor = CSP(n_components=k, reg=self.reg, log=True, norm_trace=True)
        self.csp_lateral = CSP(n_components=k, reg=self.reg, log=True, norm_trace=True)

    @staticmethod
    def posto_efetivo(X: np.ndarray, tol: float = 1e-9) -> int:
        """Posto numérico da covariância média entre canais.

        Sob CAR o posto é ``n_canais - 1`` por construção: a referência média comum
        zera a direção de modo comum, e essa direção continua nula depois de qualquer
        filtro temporal aplicado igualmente a todos os canais. Medido em vez de
        assumido porque o pipeline pode entregar dados sem CAR.
        """
        amostra = X[:: max(1, len(X) // 64)]
        C = np.mean([np.cov(x) for x in amostra], axis=0)
        w = np.linalg.eigvalsh(C)
        return int(np.sum(w > tol * max(w.max(), np.finfo(float).tiny)))

    def _ajusta_capacidade(self, X: np.ndarray) -> None:
        """Limita o número de filtros CSP ao posto disponível (correção D-3).

        Com ``n_components`` maior que o posto, o CSP devolve direções cuja variância
        é zero a menos de arredondamento, e como ``log=True`` a feature entregue ao
        LDA passa a ser o log do erro de ponto flutuante. O defeito não levanta
        exceção e não emite aviso; ele apenas cresce conforme a montagem encolhe.
        Medido antes da correção, a fração da decisão do LDA governada por features
        degeneradas ia de 1,3% em 64 canais a 26,9% em 4 — ou seja, correlacionada
        com o próprio eixo que o benchmark mede, que é a pior forma possível de um
        artefato aparecer num estudo de densidade.

        Reduzir a capacidade faz o modelo ter menos filtros em montagens pequenas.
        Isso é real e não um efeito colateral: não existem seis filtros espaciais
        independentes em dois eletrodos. ``n_components_efetivo`` fica exposto para
        que o harness registre o valor no CSV em vez de deixá-lo implícito.
        """
        k = max(1, min(self.n_components, self.posto_efetivo(X)))
        if k != self.n_components_efetivo:
            self.n_components_efetivo = k
            self._monta_csp(k)

    def fit_phase1_real(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalCSPLDASetup1V1":
        """Ajusta ambos os CSP e ambos os LDA em dados de **execução real** (rótulos 0/1/2)."""
        y = np.asarray(y).astype(np.int64).ravel()
        if len(np.unique(y)) < 2:
            raise ValueError("Fase 1: são necessárias pelo menos 2 classes em y.")
        self._ajusta_capacidade(X)

        y_motor = (y > 0).astype(np.int64)
        self.csp_motor.fit(X, y_motor)
        Xm = self.csp_motor.transform(X)
        self.lda_motor.fit(Xm, y_motor)

        mask = y > 0
        if int(np.sum(mask)) < max(4, 2 * self.n_components_efetivo):
            raise ValueError(
                "Fase 1: poucas épocas motora/lateral para CSP (após filtros). "
                f"N={int(np.sum(mask))}."
            )
        X_lat = X[mask]
        y_lat = (y[mask] - 1).astype(np.int64)
        self.csp_lateral.fit(X_lat, y_lat)
        Xlf = self.csp_lateral.transform(X_lat)
        self.lda_lateral.fit(Xlf, y_lat)
        self._phase1_fitted = True
        return self

    def fit_phase2_mi(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalCSPLDASetup1V1":
        """
        *Fine-tuning* em MI: transforma com os **CSP da Fase 1** e reajusta **só os LDA**.
        """
        if not self._phase1_fitted:
            raise RuntimeError("Chame fit_phase1_real antes de fit_phase2_mi.")

        y = np.asarray(y).astype(np.int64).ravel()
        y_motor = (y > 0).astype(np.int64)
        Xm = self.csp_motor.transform(X)
        self.lda_motor.fit(Xm, y_motor)

        mask = y > 0
        if int(np.sum(mask)) < 2:
            raise ValueError("Fase 2: sem épocas MI lateral (y>0) para reajustar LDA lateral.")
        X_lat = X[mask]
        y_lat = (y[mask] - 1).astype(np.int64)
        Xlf = self.csp_lateral.transform(X_lat)
        self.lda_lateral.fit(Xlf, y_lat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predição em cascata: saída em ``{0,1,2}``."""
        Xm = self.csp_motor.transform(X)
        motor = self.lda_motor.predict(Xm).astype(np.int64)

        out = np.zeros(len(X), dtype=np.int64)
        rest = motor == 0
        out[rest] = 0

        mot_idx = np.where(~rest)[0]
        if mot_idx.size > 0:
            Xlf = self.csp_lateral.transform(X[mot_idx])
            lat = self.lda_lateral.predict(Xlf).astype(np.int64)
            out[mot_idx] = 1 + lat
        return out

    def predict_proba_three_class(self, X: np.ndarray) -> np.ndarray:
        """
        Probabilidades 3 classes: ``P(0)=P(repouso)``, ``P(1/2)=P(motor)·P(esq/dir|·)`` com
        ``P(lat|·)`` do LDA lateral aplicado a todas as amostras (após CSP lateral).
        """
        Xm = self.csp_motor.transform(X)
        p_rm = self.lda_motor.predict_proba(Xm)
        i0 = int(np.where(self.lda_motor.classes_ == 0)[0][0])
        i1 = int(np.where(self.lda_motor.classes_ == 1)[0][0])
        p_rest = p_rm[:, i0]
        p_motor = p_rm[:, i1]

        Xlf = self.csp_lateral.transform(X)
        p_lat = self.lda_lateral.predict_proba(Xlf)
        j0 = int(np.where(self.lda_lateral.classes_ == 0)[0][0])
        j1 = int(np.where(self.lda_lateral.classes_ == 1)[0][0])

        n = len(X)
        proba = np.zeros((n, 3), dtype=np.float64)
        proba[:, 0] = p_rest
        proba[:, 1] = p_motor * p_lat[:, j0]
        proba[:, 2] = p_motor * p_lat[:, j1]

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba /= row_sums
        return proba

    # Correcao B-4: saida macia e requisito de admissao na tabela primaria, e o
    # harness a procura pelo nome canonico `predict_proba`. Sem este alias a chamada
    # levantava AttributeError, o harness capturava, gravava probabilidade vazia, e o
    # modelo saia da tabela sem que nada falhasse de forma visivel.
    predict_proba = predict_proba_three_class


def create_csp_lda_setup1_v1_model(
    n_components: int = 6,
    reg: str = "ledoit_wolf",
    lda_solver: str = "svd",
) -> HierarchicalCSPLDASetup1V1:
    """Fábrica para o baseline CSP+LDA hierárquico Setup 1 (v1)."""
    return HierarchicalCSPLDASetup1V1(n_components=n_components, reg=reg, lda_solver=lda_solver)


def create_csp_lda_model(n_components: int = 6) -> CSPLDAClassifier:
    """
    Factory function to create a fresh CSP+LDA model.
    
    This is used by evaluation functions that need to create
    new model instances for each fold/subject.
    
    Parameters
    ----------
    n_components : int
        Number of CSP components
        
    Returns
    -------
    model : CSPLDAClassifier
        Fresh model instance
    """
    return CSPLDAClassifier(n_components=n_components)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from core.data_loader import load_subject
    from core.preprocessing import preprocess_epochs
    from core.metrics import compute_metrics, format_metrics
    from sklearn.model_selection import train_test_split
    
    print("Testing CSP+LDA baseline on Subject 1...")
    
    # Load and preprocess
    X, y = load_subject(1)
    X = preprocess_epochs(X, filter_data=True)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train and evaluate
    model = create_csp_lda_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    print(f"\nResults: {format_metrics(metrics)}")
    
    print(f"\nCSP patterns shape: {model.get_csp_patterns().shape}")
