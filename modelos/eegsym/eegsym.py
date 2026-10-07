"""
EEGSym - CNN with Hemispheric Symmetry for Inter-Subject MI Classification

Exploits the natural symmetry of the brain between left and right hemispheres
for improved inter-subject generalization.

Reference: EEGSym: Overcoming Inter-Subject Variability in Motor Imagery Based BCIs

Paper Parameters:
- Channels: 8 (F3, C3, P3, Cz, Pz, F4, C4, P4) or 16
- Preprocessing: Notch 50Hz, bandpass (implicit in model)
- Data Augmentation: Patch perturbation, hemisphere flip, random shift
- Architecture: Symmetric CNN exploiting left-right brain symmetry
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
from typing import Optional, List, Tuple

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
from core.preprocessing import common_average_reference


# 8-channel configuration (paper's minimal setup)
CHANNELS_8 = ['F3', 'C3', 'P3', 'Cz', 'Pz', 'F4', 'C4', 'P4']

# 16-channel configuration
CHANNELS_16 = [
    'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

# PhysioNet indices for 8-channel subset (approximate based on 10-10 system)
# F3≈4, C3≈13, P3≈23, Cz≈14, Pz≈24, F4≈6, C4≈15, P4≈25
PHYSIONET_8CH_INDICES = [4, 13, 23, 14, 24, 6, 15, 25]

# Left and right hemisphere channel groups (for symmetry)
LEFT_CHANNELS = [0, 1, 2]  # F3, C3, P3 in 8-ch config
RIGHT_CHANNELS = [5, 6, 7]  # F4, C4, P4 in 8-ch config
CENTER_CHANNELS = [3, 4]  # Cz, Pz in 8-ch config

# Disposicao herdada do artigo: [E, E, E, C, C, D, D, D]. Usada apenas quando o
# chamador NAO informa os nomes dos canais, para preservar o comportamento antigo.
LAYOUT_ARTIGO = ([0, 1, 2, 3, 4], [5, 6, 7, 3, 4])


def espelho(ch: str) -> Optional[str]:
    """Nome do eletrodo espelhado no outro hemisferio, ou None se for linha media.

    A nomenclatura 10-05 codifica o hemisferio no sufixo numerico: impar a
    esquerda, par a direita, `z` na linha media. O espelho de um impar n e n+1
    (C3 -> C4, FC5 -> FC6, PO7 -> PO8). O prefixo nunca contem digitos, entao
    separar pelo sufixo e seguro.
    """
    d = "".join(c for c in ch if c.isdigit())
    if not d:
        return None
    n = int(d)
    pref = ch[:len(ch) - len(d)]
    return f"{pref}{n + 1}" if n % 2 else f"{pref}{n - 1}"


def divide_hemisferios(ch_names: List[str]):
    """Monta os dois ramos do EEGSym a partir dos NOMES dos canais.

    A arquitetura passa dois ramos por um encoder partilhado (Siamese) e so faz
    sentido se a posicao k de um ramo for o eletrodo espelhado da posicao k do
    outro. Indices fixos nao garantem isso: em `anat8`, ordenada
    [C3, C4, Cz, FC3, FC4, CP3, CP4, Fz], o recorte original entrega ao ramo
    "esquerdo" os canais C3, C4, Cz, FC3, FC4 — dois hemisferios e a linha media
    misturados, o que destroi a premissa do modelo.

    Devolve (idx_esq, idx_dir, idx_centro, descartados), com idx_esq e idx_dir
    pareados posicao a posicao. Laterais sem par sao descartados e reportados: o
    ramo precisa de tamanho igual dos dois lados, e inventar um canal ausente
    seria pior que omitir um presente.
    """
    pos = {c: i for i, c in enumerate(ch_names)}
    esq, dire, centro, descartados = [], [], [], []
    for c in ch_names:
        par = espelho(c)
        if par is None:
            centro.append(pos[c])
        elif par in pos:
            if int("".join(x for x in c if x.isdigit())) % 2:   # so na passagem do impar
                esq.append(pos[c]); dire.append(pos[par])
        else:
            descartados.append(c)
    return esq, dire, centro, descartados


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters, dropout):
        super().__init__()
        # K sizes: 500ms(81), 250ms(41), 125ms(19) at 160Hz
        self.b1 = nn.Conv2d(in_channels, n_filters, (1, 81), padding=(0, 40), bias=False)
        self.b2 = nn.Conv2d(in_channels, n_filters, (1, 41), padding=(0, 20), bias=False)
        self.b3 = nn.Conv2d(in_channels, n_filters, (1, 19), padding=(0, 9), bias=False)
        self.bn = nn.BatchNorm2d(n_filters * 3)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv2d(in_channels, n_filters * 3, (1, 1)) if in_channels != n_filters * 3 else nn.Identity()

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.dropout(self.elu(self.bn(out)))
        return out + self.proj(x)


class EEGSymEncoder(nn.Module):
    """
    Blocos convolucionais partilhados (Siamese + Inception) até ao vector de características.
    """

    def __init__(
        self,
        n_samples: int = 480,
        n_filters: int = 8,
        dropout: float = 0.4,
        layout: Optional[Tuple[List[int], List[int]]] = None,
        n_channels: int = 8,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_filters = n_filters
        # layout = (indices do ramo esquerdo, indices do ramo direito), ja com a
        # linha media anexada aos dois. Sem ele, cai na disposicao do artigo.
        self.idx_l, self.idx_r = layout if layout is not None else LAYOUT_ARTIGO
        if len(self.idx_l) != len(self.idx_r):
            raise ValueError(f"ramos de tamanhos diferentes: {len(self.idx_l)} "
                             f"e {len(self.idx_r)} — o encoder e partilhado")
        self.n_channels = n_channels
        alt = len(self.idx_l)
        self.inception = InceptionBlock(
            in_channels=1, n_filters=n_filters, dropout=dropout
        )
        self.pool1 = nn.AvgPool2d((1, 4))
        # O kernel espacial precisa colapsar o ramo inteiro a altura 1: so assim o
        # `cat` dos dois ramos da altura 2 e o `channel_merge` (2,1) fecha em 1.
        # Fixo em 5, so funcionava para o ramo de 3+2 do artigo.
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters * 3, n_filters * 6, (alt, 1), bias=False),
            nn.BatchNorm2d(n_filters * 6),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.channel_merge = nn.Sequential(
            nn.Conv2d(n_filters * 6, n_filters * 6, (2, 1), bias=False),
            nn.BatchNorm2d(n_filters * 6),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.temporal_merge = nn.Sequential(
            nn.Conv2d(n_filters * 6, n_filters * 6, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(n_filters * 6),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.feature_size = self._infer_feature_size()

    def _infer_feature_size(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            return self.forward(x).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x_left = x[:, :, self.idx_l, :]
        x_right = x[:, :, self.idx_r, :]
        l = self.spatial(self.pool1(self.inception(x_left)))
        r = self.spatial(self.pool1(self.inception(x_right)))
        merged = torch.cat([l, r], dim=2)
        out = self.temporal_merge(self.channel_merge(merged))
        return out.view(out.size(0), -1)


class EEGSymModel(nn.Module):
    """
    EEGSym v0.4 architecture (Symmetric Siamese + Inception).
    Reconstructed from Pérez-Velasco et al. (2021).
    """

    def __init__(
        self,
        n_samples: int = 480,
        n_classes: int = 2,
        n_filters: int = 8,
        dropout: float = 0.4,
        layout: Optional[Tuple[List[int], List[int]]] = None,
        n_channels: int = 8,
    ):
        super(EEGSymModel, self).__init__()
        self.n_samples = n_samples
        self.n_filters = n_filters
        self.encoder = EEGSymEncoder(
            n_samples=n_samples, n_filters=n_filters, dropout=dropout,
            layout=layout, n_channels=n_channels,
        )
        fs = self.encoder.feature_size
        self.classifier = nn.Sequential(
            nn.Linear(fs, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class EEGSymSetup1HierModel(nn.Module):
    """
    Setup 1 (planning/07): backbone EEGSym partilhado + cabeça motor (repouso vs. motora)
    + cabeça lateral (esq. vs. dir.), alinhado a EEGNet/CSP/FBCSP v1.
    """

    def __init__(
        self,
        n_samples: int = 480,
        n_filters: int = 8,
        dropout: float = 0.4,
        layout: Optional[Tuple[List[int], List[int]]] = None,
        n_channels: int = 8,
    ):
        super().__init__()
        self.encoder = EEGSymEncoder(
            n_samples=n_samples, n_filters=n_filters, dropout=dropout,
            layout=layout, n_channels=n_channels,
        )
        fs = self.encoder.feature_size

        def mlp_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fs, 64),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

        self.fc_motor = mlp_head()
        self.fc_lat = mlp_head()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x)
        return self.fc_motor(f), self.fc_lat(f)


class EEGSymClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible EEGSym classifier.
    
    Parameters
    ----------
    n_filters : int
        Number of filters in first layer
    dropout : float
        Dropout rate
    epochs : int
        Training epochs
    lr : float
        Learning rate
    batch_size : int
        Batch size
    use_8ch : bool
        Use 8-channel subset (paper's config)
    augment : bool
        Use data augmentation
    filter_band : tuple
        Bandpass filter
    sfreq : float
        Sampling frequency
    """
    
    def __init__(
        self,
        n_filters: int = 8,
        dropout: float = 0.4,
        epochs: int = 150,
        lr: float = 0.001,
        batch_size: int = 16,
        use_8ch: bool = True,
        augment: bool = False,
        filter_band: Optional[Tuple[float, float]] = (4, 40),
        sfreq: float = 160.0,
        patience: int = 20,
        val_split: float = 0.15,
        weight_decay: float = 0.0,
    ):
        self.n_filters = n_filters
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.use_8ch = use_8ch
        self.augment = augment
        self.filter_band = filter_band
        self.sfreq = sfreq
        self.patience = patience
        self.val_split = val_split
        self.label_smoothing = 0.0
        self.weight_decay = weight_decay
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_ = None

        # Nomes dos canais de entrada, informados por set_channels(). Enquanto forem
        # None o modelo assume a disposicao do artigo, que so vale se a entrada
        # estiver ordenada [E, E, E, C, C, D, D, D].
        self.ch_names = None
        self._layout = None
        self._pares = None
        self.descartados = []

        # Setup filter
        self.filter_b = None
        self.filter_a = None
        if self.filter_band is not None:
            nyq = self.sfreq / 2.0
            low = self.filter_band[0] / nyq
            high = self.filter_band[1] / nyq
            self.filter_b, self.filter_a = signal.butter(5, [low, high], btype='band')
    
    def set_channels(self, ch_names: List[str]) -> "EEGSymClassifier":
        """Informa os nomes dos canais de entrada, na ordem em que chegam em X.

        Sem isto o modelo recorta os hemisferios por indices fixos, o que so vale
        para a ordem do artigo. Com montagens arbitrarias os dois ramos acabam
        recebendo canais dos dois lados, e a simetria que da nome a arquitetura
        deixa de existir.

        Chamar antes de fit(). O encoder e construido em fit(), entao aqui so se
        calcula e guarda a disposicao.
        """
        esq, dire, centro, desc = divide_hemisferios(list(ch_names))
        if not esq:
            raise ValueError(
                f"nenhum par esquerda/direita em {list(ch_names)} — o EEGSym exige "
                f"pelo menos um par espelhado para formar os dois ramos")
        self.ch_names = list(ch_names)
        # Linha media vai aos DOIS ramos, como no artigo: e a referencia comum que
        # permite ao encoder partilhado comparar os hemisferios.
        self._layout = (esq + centro, dire + centro)
        self._pares = list(zip(esq, dire))
        self.descartados = desc
        return self

    def _confere_entrada(self, n_channels: int) -> None:
        """Aborta se a disposicao nao corresponder a largura real de X.

        Uma discrepancia aqui significa que set_channels() recebeu uma montagem e o
        loader entregou outra. Sem esta checagem o modelo indexaria posicoes validas
        mas de canais errados, e treinaria sem erro nenhum — que e exatamente como o
        recorte por indices fixos passou despercebido ate agora.
        """
        if self._layout is None:
            if n_channels != 8:
                raise ValueError(
                    f"entrada com {n_channels} canais sem set_channels(). Sem os nomes "
                    f"o EEGSym so sabe recortar hemisferios em 8 canais na ordem do "
                    f"artigo [E,E,E,C,C,D,D,D].")
            return
        if len(self.ch_names) != n_channels:
            raise ValueError(
                f"set_channels() recebeu {len(self.ch_names)} nomes mas X traz "
                f"{n_channels} canais — montagem e dados divergem")

    @property
    def n_canais_efetivo(self) -> int:
        """Canais que o modelo de fato usa, depois do descarte de laterais sem par.

        Correcao D-5. A arquitetura precisa de pares espelhados; um canal lateral sem
        par nao entra em nenhum dos dois ramos e e descartado em silencio. Nas duas
        montagens assimetricas aceitas isso significa que a celula rotulada "21
        canais" usa 20, e a "sem_fz, 8 canais" usa 7. Sem este valor no CSV, uma
        curva de desempenho por densidade colocaria o EEGSym na abscissa errada, e o
        contraste `sem_fz` contra `anat8` confundiria "sem Fz" com "um canal a menos".
        """
        if self._layout is None:
            return len(self.ch_names) if self.ch_names else 0
        return len(self.ch_names) - len(self.descartados)

    def layout_legivel(self) -> str:
        """Descreve os dois ramos por nome, para registro no log da run."""
        if self._layout is None:
            return "disposicao do artigo (indices fixos; entrada precisa ser [EEE CC DDD])"
        n = self.ch_names
        l, r = self._layout
        s = (f"ramo E: {' '.join(n[i] for i in l)}\n"
             f"      ramo D: {' '.join(n[i] for i in r)}")
        if self.descartados:
            s += f"\n      descartados (lateral sem par): {' '.join(self.descartados)}"
        return s

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """Preprocess: CAR + channel subset + bandpass filter."""
        # Step 0: Common Average Reference (remove shared noise)
        X = common_average_reference(X)

        # Recorte de canais. Com set_channels() nao ha recorte nenhum: a disposicao
        # indexa as posicoes reais de X e os canais sem par simplesmente nunca sao
        # selecionados no forward. O recorte por PHYSIONET_8CH_INDICES abaixo so
        # sobrevive para chamadores antigos, e e justamente ele que fazia o modelo
        # ver 8 canais fixos mesmo recebendo 16, 21, 32 ou 64 — apagando o eixo de
        # densidade que o benchmark existe para medir.
        if self._layout is None and self.use_8ch and X.shape[1] > 8:
            # Use approximate motor cortex channels
            valid_indices = [i for i in PHYSIONET_8CH_INDICES if i < X.shape[1]]
            if len(valid_indices) >= 8:
                X = X[:, valid_indices[:8], :]
            else:
                # Fallback: use first 8 channels
                X = X[:, :8, :]

        # Bandpass filter
        if self.filter_b is not None:
            X_filtered = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_filtered[i] = signal.filtfilt(self.filter_b, self.filter_a, X[i], axis=1)
            X = X_filtered
        
        return X
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        if not self.augment:
            return X, y
        
        augmented_X = [X]
        augmented_y = [y]
        
        # Hemisphere flip augmentation
        # Swap left and right channels, flip labels
        # Os pares vem de set_channels() quando disponivel; os indices fixos abaixo
        # so valem para a ordem do artigo e trocariam canais errados em qualquer
        # outra montagem.
        pares = getattr(self, "_pares", None)
        if pares is None and X.shape[1] == 8:
            pares = [(0, 5), (1, 6), (2, 7)]   # F3/F4, C3/C4, P3/P4
        if pares:
            X_flipped = X.copy()
            e = [a for a, _ in pares]
            d = [b for _, b in pares]
            X_flipped[:, e, :] = X[:, d, :]
            X_flipped[:, d, :] = X[:, e, :]
            y_flipped = 1 - y  # Flip labels
            augmented_X.append(X_flipped)
            augmented_y.append(y_flipped)

        return np.vstack(augmented_X), np.hstack(augmented_y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model with early stopping and LR scheduling."""
        self.classes_ = np.unique(y)
        
        # Preprocess
        X = self._preprocess(X)
        
        # Augment (disabled by default in v0.2)
        X, y = self._augment_data(X, y)
        
        # Get dimensions
        n_channels = X.shape[1]
        n_samples = X.shape[2]
        n_classes = len(self.classes_)
        
        # Create model (v0.4 Architecture)
        self._confere_entrada(n_channels)
        self.model = EEGSymModel(
            n_samples=n_samples,
            n_classes=n_classes,
            n_filters=self.n_filters,
            dropout=self.dropout,
            layout=self._layout,
            n_channels=n_channels,
        ).to(self.device)
        
        # Train/val split for early stopping
        n_val = max(1, int(len(X) * self.val_split))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]
        
        X_train_t = torch.FloatTensor(X[train_idx]).to(self.device)
        y_train_t = torch.LongTensor(y[train_idx]).to(self.device)
        X_val_t = torch.FloatTensor(X[val_idx]).to(self.device)
        y_val_t = torch.LongTensor(y[val_idx]).to(self.device)
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size, shuffle=True
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        best_weights = copy.deepcopy(self.model.state_dict())
        no_improve = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()
            
            scheduler.step(val_loss)
            
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


class EEGSymSetup1Classifier(EEGSymClassifier):
    """
    EEGSym para **Setup 1** (planning/07): backbone partilhado + cabeça **motor**
    (repouso vs. motora) + cabeça **lateral** (esq. vs. dir.), com treino em duas fases
    via ``fit`` + ``continue_training``.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGSymSetup1Classifier":
        self.classes_ = np.array([0, 1, 2])
        X = self._preprocess(X)
        y = np.asarray(y).astype(np.int64).ravel()
        n_samples = X.shape[2]
        self._confere_entrada(X.shape[1])
        self.model = EEGSymSetup1HierModel(
            n_samples=n_samples,
            n_filters=self.n_filters,
            dropout=self.dropout,
            layout=self._layout,
            n_channels=X.shape[1],
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
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                om, ol = self.model(batch_X)
                loss = self._loss_setup1(om, ol, batch_y, crit)
                loss.backward()
                optimizer.step()

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
    ) -> "EEGSymSetup1Classifier":
        if self.model is None:
            raise RuntimeError("continue_training requires fit() first")
        if not isinstance(self.model, EEGSymSetup1HierModel):
            raise TypeError("continue_training Setup 1 espera EEGSymSetup1HierModel")

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
        optimizer = optim.Adam(
            self.model.parameters(), lr=train_lr, weight_decay=self.weight_decay
        )

        best_val = float("inf")
        best_w = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for _ in range(epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                om, ol = self.model(batch_X)
                loss = self._loss_setup1(om, ol, batch_y, crit)
                loss.backward()
                optimizer.step()

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


def create_eegsym_setup1_model(
    n_filters: int = 8,
    dropout: float = 0.4,
    epochs: int = 150,
    lr: float = 0.001,
    batch_size: int = 16,
    patience: int = 20,
    weight_decay: float = 1e-4,
    filter_band: Optional[Tuple[float, float]] = (4, 40),
    val_split: float = 0.15,
) -> EEGSymSetup1Classifier:
    """EEGSym no desenho Setup 1 (duas cabeças); sem *data augmentation* (rótulos 0/1/2).

    ``filter_band`` e ``val_split`` viraram parâmetros pela correção B-6; ver a nota
    equivalente em ``eegnet.create_eegnet_setup1_v7_model``.
    """
    return EEGSymSetup1Classifier(
        n_filters=n_filters,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_8ch=True,
        augment=False,
        patience=patience,
        weight_decay=weight_decay,
        filter_band=filter_band,
        val_split=val_split,
        sfreq=160.0,
    )


def create_eegsym_model(
    n_filters: int = 8,
    dropout: float = 0.4,
    epochs: int = 150,
    lr: float = 0.001,
    batch_size: int = 16,
    patience: int = 20
) -> EEGSymClassifier:
    """
    Factory function to create EEGSym classifier.
    
    Returns
    -------
    model : EEGSymClassifier
        Fresh model instance
    """
    return EEGSymClassifier(
        n_filters=n_filters,
        dropout=dropout,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_8ch=True,
        augment=False,
        patience=patience,
        filter_band=(4, 40),
        sfreq=160.0
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from core.data_loader import load_subject
    from core.metrics import compute_metrics, format_metrics
    from sklearn.model_selection import train_test_split
    
    print("Testing EEGSym on Subject 1...")
    
    # Load data
    X, y = load_subject(1)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create and train model
    model = create_eegsym_model(epochs=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    print(f"\nResults: {format_metrics(metrics)}")
