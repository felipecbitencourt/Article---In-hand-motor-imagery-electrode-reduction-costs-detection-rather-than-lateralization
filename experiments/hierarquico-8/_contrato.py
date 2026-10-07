# -*- coding: utf-8 -*-
"""Contrato de harmonizacao do benchmark por montagem.

Implementa as dez correcoes bloqueantes de PROPOSTA_BENCHMARK.md secao 3 e o
contrato da secao 4. Tres responsabilidades, nesta ordem de importancia:

1. DONO UNICO DO PRE-PROCESSAMENTO (B-1, B-5, B-6, B-7, B-9). `prepara` e o unico
   lugar do pipeline onde canais sao selecionados, referencia e aplicada, banda e
   filtrada, epocas sao rejeitadas e a escala e definida. Nenhum modelo repete
   nenhuma dessas operacoes, e cada modelo e CONSTRUIDO de forma a nao poder
   repeti-la (`filter_band=None`, `use_channel_subset=False`).

2. DECLARACAO VERIFICAVEL (B-2, B-4, B-10). O que e uniforme esta em `CONTRATO`;
   o que diverge esta em `MODELOS[nome]["divergencias"]` com quatro campos
   obrigatorios. `portao_declaracao` roda antes da primeira epoca ser carregada e
   aborta a run por omissao, nao por comissao: modelo sem entrada nao roda.

3. CARIMBO (B-10). `carimbo` resume as condicoes num hash de 10 caracteres que vai
   em toda linha do CSV, ao lado de propriedades MEDIDAS na propria run. Duas
   linhas com hashes diferentes nao pertencem a mesma tabela, e `recusa_mistura`
   existe para que o agregador nao possa junta-las por descuido.

POR QUE UM MODULO SEPARADO E NAO UMA EMENDA EM run_all_models.py

O registro de `run_all_models.py` esta duplicado em quatro experimentos
(hierarquico-8, -21, -64, imageamento-64) e o `main()` de cada copia alimenta o
pipeline antigo, que produziu a Tabela 1 do artigo submetido. Remover dali o campo
`preprocess_fn`, como pedia a redacao original de B-6, apagaria a referencia contra
a qual o pipeline novo precisa ser comparado. O contrato vive aqui, o pipeline
antigo continua reproduzivel la, e a separacao e o que permite medir a diferenca
entre os dois em vez de perde-la.

TRES ACHADOS QUE CORRIGEM A PROPOSTA

- FBCSP nao precisa de celula gemea de banda. A proposta apontou o banco de 8 a 32
  Hz de `fbcsp_svm.py:56-64`, que pertence a classe NAO hierarquica. A classe do
  Setup 1 usa `DEFAULT_SETUP1_FREQ_BANDS` (`:306-316`), nove sub-bandas de 4 a 40,
  cuja uniao ja e exatamente a banda do contrato.
- Todos os oito modelos aplicam CAR internamente. O CAR e idempotente (apos o
  primeiro, a media entre canais e zero por construcao, e o segundo subtrai zero),
  entao o CAR do harness domina: aplicado APOS a selecao, ele fixa a referencia
  sobre a montagem, e a repeticao interna do modelo vira no-op. E por isso que o
  braco sem CAR ainda precisa do patch de modulo — sem ele os modelos religariam a
  referencia que o braco existe para desligar.
- A validacao interna e 15%, nao 20% como dizia a proposta. Os cinco profundos
  compartilham `val_split=0.15`; o valor uniforme declarado e o que os cinco de
  fato usam, e agora ele e passado explicitamente em vez de herdado por default.
"""
from __future__ import annotations

import functools
import hashlib
import json
import inspect
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
#  1. O que e uniforme (secao 4.1)
# ---------------------------------------------------------------------------

CONTRATO = {
    # sinal
    "escala": "microvolts",
    "janela_s": (0.5, 2.5),
    "sfreq": 160.0,
    "banda_hz": (4.0, 40.0),
    "filtro": "butter_filtfilt",
    "ordem_filtro": 5,
    "referencia": "car_apos_selecao",
    "car": True,
    # rejeicao
    "rejeicao": "iqr_tukey_maxabs",
    "iqr_k": 1.5,
    "rejeicao_escopo": "sujeito_x_fase",
    "trava_min_keep": 0.5,          # condicao de erro, nunca criterio
    # protocolo
    "calib_n": 48,
    "folds": 3,
    # Fases de treino do pool. "real+mi" e o desenho original, com movimento
    # executado antes do imageamento; "mi" treina uma vez so, sobre imageamento,
    # e responde se a fase de movimento real contribui de fato. Vai ao carimbo:
    # os dois bracos medem coisas diferentes e nao podem entrar na mesma tabela.
    "fases_treino": "real+mi",
    "fase2_regime": "total",
    "unidade_estatistica": "sujeito",
    # treino
    "teto_epocas": 200,
    "paciencia": 20,
    "val_split": 0.15,
    "otimizador": "adam",
    "weight_decay": 1e-4,
    "lr": 1e-3,
    "batch_size": 16,
    "calib_epocas": 80,
    "calib_lr": 1e-4,
    "calib_paciencia": 15,
}

# Campos que nenhum modelo pode redefinir. Usado pelo portao de declaracao.
UNIFORMES_PROIBIDOS = frozenset({
    "escala", "banda_hz", "ordem_filtro", "referencia", "car", "rejeicao",
    "iqr_k", "sfreq", "calib_n", "folds", "otimizador", "weight_decay",
    "teto_epocas", "paciencia", "val_split",
})


def _div(valor, motivo, fonte, gemea):
    """Uma divergencia declarada. Os quatro campos sao obrigatorios (secao 4.2)."""
    return {"valor": valor, "motivo": motivo, "fonte": fonte, "exige_gemea": gemea}


# ---------------------------------------------------------------------------
#  2. O que e por modelo (secao 4.2)
# ---------------------------------------------------------------------------

MODELOS = {
    "csp_lda": dict(
        pacote="csp_lda", familia="covariancia", dispositivo="cpu",
        divergencias={
            "n_components": _div(
                6, "numero de filtros CSP e camada livre de capacidade",
                "Ramoser 2000, secao III", False),
        },
    ),
    "fbcsp_svm": dict(
        pacote="fbcsp_svm", familia="covariancia", dispositivo="cpu",
        divergencias={
            "banco_de_bandas": _div(
                "9 sub-bandas de 4 a 40 Hz",
                "particao interna da banda e a arquitetura do metodo; a uniao das "
                "sub-bandas iguala a banda do contrato, entao nao ha divergencia "
                "de banda efetiva",
                "Ang 2008; DEFAULT_SETUP1_FREQ_BANDS em fbcsp_svm.py:306-316", False),
            "svm_kernel": _div(
                "linear", "kernel linear expoe predict_proba via SVC(probability=True); "
                "o ramo liblinear nao expoe e por isso e inadmissivel",
                "fbcsp_svm.py:365-369", False),
        },
    ),
    "riemannian": dict(
        pacote="riemannian_dcca", familia="covariancia", dispositivo="cpu",
        divergencias={
            "banda_do_artigo": _div(
                (8.0, 30.0),
                "o artigo de base especifica 8 a 30 Hz; sob o contrato o braco "
                "primario roda em 4 a 40 e a leitura fiel roda como celula gemea",
                "riemannian_dcca.py:420 (Paper uses 8-30 Hz)", True),
            "estimador_cov": _div(
                "lwf", "estimador de covariancia e camada livre", "pyriemann", False),
        },
    ),
    "eegnet": dict(
        pacote="eegnet", familia="profundo", dispositivo="gpu",
        divergencias={
            "capacidade": _div(
                "F1=16, D=4, F2=64, dropout=0.5",
                "capacidade da arquitetura nao altera condicoes de comparacao",
                "Lawhern 2018; configuracao final revisada na monografia", False),
        },
    ),
    "eegsym": dict(
        pacote="eegsym", familia="profundo", dispositivo="gpu",
        divergencias={
            "capacidade": _div(
                "n_filters=8, dropout=0.4", "camada livre", "Perez-Velasco 2022", False),
            "otimizador_nao_parametrizavel": _div(
                "adam", "o otimizador e fixo no codigo e ja coincide com o contrato, "
                "entao satisfaz a propriedade uniforme sem poder recebe-la por argumento",
                "eegsym.py:501 (optim.Adam)", False),
            "ramos_hemisfericos": _div(
                "set_channels obrigatorio",
                "a arquitetura parte a entrada em dois ramos hemisfericos; sem os "
                "nomes dos canais os ramos misturam os lados",
                "Perez-Velasco 2022, secao 2.2", False),
        },
    ),
    "shallow_convnet": dict(
        pacote="shallow_convnet", familia="profundo", dispositivo="gpu",
        divergencias={
            "capacidade": _div(
                "40 filtros, kernel 25, pool 75/15, dropout=0.5", "camada livre",
                "Schirrmeister 2017, Tabela 1", False),
        },
    ),
    "deepcnn": dict(
        pacote="cnn", familia="profundo", dispositivo="gpu",
        divergencias={
            "capacidade": _div("dropout=0.5", "camada livre", "Schirrmeister 2017", False),
        },
    ),
    "stia_net": dict(
        pacote="stia_net", familia="profundo", dispositivo="gpu",
        divergencias={
            "banda_do_artigo": _div(
                (8.0, 30.0),
                "o artigo de base especifica 8 a 30 Hz; braco primario em 4 a 40 e "
                "leitura fiel como celula gemea, igual ao riemannian",
                "run_all_models.py:222 (filter_band=(8, 30))", True),
            "capacidade": _div(
                "gcn 64/32, tcn (16,32,64), 8 cabecas, dropout=0.5", "camada livre",
                "configuracao final revisada na monografia", False),
        },
    ),
}


# ---------------------------------------------------------------------------
#  3. Pre-processamento de dono unico (B-1, B-5, B-6, B-7, B-9)
# ---------------------------------------------------------------------------

class RejeicaoExcessiva(RuntimeError):
    """A trava de retencao disparou.

    Correcao B-7. Na versao anterior a trava era um CRITERIO: quando o IQR (ou o
    limiar fixo de 100 uV) rejeitava mais que a fracao permitida, a funcao mantinha
    silenciosamente as N melhores epocas, e o criterio efetivo virava "as 50% de
    menor amplitude" sem que nada no log dissesse isso. Medido no pipeline do
    artigo: disparava em 26 de 30 alvos. Aqui ela e uma condicao de erro que
    sinaliza a celula como suspeita; com IQR de Tukey ela nunca deveria disparar, e
    se disparar o dado e que esta errado, nao o criterio.
    """


def _car(X: np.ndarray) -> np.ndarray:
    """Referencia media comum sobre os canais PRESENTES no array recebido.

    Aplicada apos a selecao de canais, de proposito: um dispositivo real de 8
    eletrodos teria a referencia igualmente pobre. Calcula-la sobre os 64 e depois
    recortar injetaria nos 8 usados informacao dos 56 descartados.
    """
    return X - X.mean(axis=1, keepdims=True)


def _passa_banda(X: np.ndarray, sfreq: float, banda, ordem: int) -> np.ndarray:
    """Butterworth + filtfilt, com sfreq EXPLICITA e ordem explicita.

    Correcao B-1. A chamada anterior usava nomes de argumento que nao existiam na
    assinatura de `core.preprocessing.preprocess_epochs` (`sfreq=`, `l_freq=`,
    `h_freq=`, `apply_car=`), e um `except TypeError` do proprio harness engolia o
    erro e recaia em `preprocess_epochs(Xs, 160.0)` — onde 160.0 chegava no
    parametro `filter_data`, e a banda usada era o default de 8 a 30 Hz, nao os 4 a
    40 declarados. Nenhuma string estava errada; o defeito so aparecia rodando.
    Aqui nao ha fallback: se a chamada estiver errada, ela levanta.
    """
    nyq = sfreq / 2.0
    b, a = signal.butter(ordem, [banda[0] / nyq, banda[1] / nyq], btype="band")
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = signal.filtfilt(b, a, X[i], axis=1)
    return out


def _decide_iqr(X: np.ndarray, k: float, trava: float, rotulo: str) -> np.ndarray:
    """IQR de Tukey sobre o maximo absoluto por epoca. Devolve os INDICES mantidos.

    Reimplementado aqui, e nao herdado de `core.preprocessing.reject_epochs_iqr`,
    por duas razoes que sao o proprio ponto do contrato: o harness precisa ser o
    dono do criterio, e a trava precisa levantar em vez de silenciar (B-7). O
    criterio numerico e identico. E invariante de escala — o `*1e6` da versao do
    core nao muda quais epocas passam, so o limiar impresso.
    """
    amp = np.max(np.abs(X), axis=(1, 2))
    q1, q3 = np.percentile(amp, [25, 75])
    limiar = q3 + k * (q3 - q1)
    keep = np.where(amp < limiar)[0]
    if len(keep) < trava * len(X):
        raise RejeicaoExcessiva(
            f"{rotulo}: IQR manteria {len(keep)}/{len(X)} epocas "
            f"({100*len(keep)/max(len(X),1):.0f}%), abaixo da trava de "
            f"{100*trava:.0f}%. Celula suspeita — o dado, nao o criterio.")
    return keep


def prepara(X: np.ndarray, idx, *, car: bool, sfreq: float, banda, ordem: int,
            iqr_k: float, trava: float, rotulo: str = ""):
    """Selecao -> referencia -> banda -> rejeicao -> escala. Nesta ordem, uma vez so.

    Devolve `(X_uv, keep)`, onde `X_uv` esta em microvolts e ja filtrado — e o sinal
    que o modelo recebe, sem nenhum passo pendente — e `keep` sao os indices das
    epocas mantidas DENTRO do array original.

    Sobre `keep` (correcao B-3): a versao anterior devolvia so o sinal, e o
    identificador gravado no CSV de predicoes era a posicao dentro do array JA
    filtrado. Como a rejeicao passou a ser calculada depois da selecao de canais,
    ela retem conjuntos diferentes em cada densidade, e a mesma posicao passou a
    apontar para epocas diferentes em 8, 21 e 64 canais — medido: os rotulos diferem
    em 53% a 60% das posicoes entre densidades. Qualquer analise pareada por epoca
    (matriz de decisao, ortogonalidade entre modelos) estava comparando epocas que
    nao eram a mesma. `keep[j]` e estavel porque indexa o array original.

    Sobre `car` (correcao B-9): a decisao de rejeicao e SEMPRE calculada com CAR,
    independente do braco. Se o braco sem CAR decidisse sobre o sinal sem CAR, ele
    mudaria tambem QUAIS epocas sobrevivem, e a diferenca medida entre os bracos
    misturaria o efeito da referencia com o efeito da retencao. Fixando a decisao, o
    unico contraste entre os bracos e o CAR em si.

    Sobre a escala (correcao B-5): microvolts, aplicados uma vez, aqui. Modelos sem
    normalizacao na primeira camada recebiam volts (ordem de 1e-5) e operavam na
    faixa onde a saida colapsa numa classe so. Nenhum modelo reescala de novo —
    verificado: nao ha `1e6` em nenhum arquivo de `modelos/`.
    """
    Xs = (X[:, idx, :] if idx is not None else X).astype(np.float64)
    Xd = _passa_banda(_car(Xs), sfreq, banda, ordem)          # decisao: sempre com CAR
    keep = _decide_iqr(Xd, iqr_k, trava, rotulo)
    Xe = Xd if car else _passa_banda(Xs, sfreq, banda, ordem)  # entrega: conforme o braco
    return Xe[keep] * 1e6, keep


def desliga_car_nos_modelos():
    """Neutraliza o CAR interno dos oito modelos, para o braco de controle.

    O CAR do harness ja e desligado pelo argumento `car=False` de `prepara`. Este
    patch existe porque os oito modelos aplicam CAR por conta propria em
    `_preprocess`, e sem ele o braco de controle nao desligaria coisa nenhuma. Tem
    de ser chamado ANTES do import dos modelos: cada um faz `from core.preprocessing
    import common_average_reference`, e a ligacao le o atributo do modulo no momento
    do import.

    A retencao de epocas nao muda com este patch, porque a decisao de rejeicao roda
    dentro de `prepara`, sempre com CAR, e nao passa por esta funcao.
    """
    import core.preprocessing as CP
    CP.common_average_reference = lambda X: X


# ---------------------------------------------------------------------------
#  4. Orcamento de calibracao (B-8)
# ---------------------------------------------------------------------------

def orcamento_calibracao(y_treino: np.ndarray, n: int, seed: int):
    """Sorteia N epocas estratificadas do lado de treino do fold.

    Correcao B-8. Antes, a calibracao usava TODO o lado de treino do fold, cujo
    tamanho depende de quantas epocas sobreviveram a rejeicao. Como a rejeicao varia
    com a densidade, "menos canais" e "menos dados de calibracao" mudavam juntos, e
    a queda medida entre 64 e 8 canais nao podia ser atribuida a nenhum dos dois. Um
    numero absoluto separa os dois efeitos: a retencao continua variando, o
    orcamento nao.

    Estratificado por alocacao proporcional com maior resto, de modo que o modelo
    veja o mesmo prior de classes que o conjunto de teste. Devolve
    `(posicoes, atingiu)`; `atingiu=False` marca a celula em que nao havia epocas
    suficientes para o orcamento, e essa celula vai ao CSV com a marca em vez de
    entrar em silencio com um orcamento menor.
    """
    total = len(y_treino)
    if n is None or n >= total:
        return np.arange(total), (n is None or n == total)

    classes, contagens = np.unique(y_treino, return_counts=True)
    exato = contagens / contagens.sum() * n
    cota = np.floor(exato).astype(int)
    sobra = n - int(cota.sum())
    if sobra > 0:
        cota[np.argsort(-(exato - cota))[:sobra]] += 1

    rng = np.random.default_rng(seed)
    sel, atingiu = [], True
    for c, q in zip(classes, cota):
        pos = np.where(y_treino == c)[0]
        if q > len(pos):
            q, atingiu = len(pos), False
        sel.append(rng.choice(pos, q, replace=False))
    return np.sort(np.concatenate(sel)), atingiu


# ---------------------------------------------------------------------------
#  5. Carimbo e propriedades medidas (B-10)
# ---------------------------------------------------------------------------

# Correcao D-6/D-7, do pre-voo. A lista omitia cinco campos do proprio CONTRATO —
# lr, batch_size, calib_epocas, calib_lr, calib_paciencia — justamente os que
# governam duas das tres fases de treino dos profundos: editar `calib_lr` no meio da
# grade produziria linhas com carimbo IDENTICO, que e o modo de falha que B-10 existe
# para impedir. E omitia semente, agrupamento e tamanho do pool, entao uma sonda com
# `--pool-max 8` carimbava igual a run completa com pool de 98.
CAMPOS_CARIMBO = ("banda_hz", "ordem_filtro", "filtro", "referencia", "car",
                  "rejeicao", "iqr_k", "rejeicao_escopo", "trava_min_keep",
                  "calib_n", "fases_treino", "fase2_regime", "escala", "sfreq", "janela_s",
                  "folds", "teto_epocas", "paciencia", "val_split",
                  "otimizador", "weight_decay",
                  "lr", "batch_size", "calib_epocas", "calib_lr", "calib_paciencia",
                  "seed", "grupos", "pool_max")


def carimbo(cfg: dict) -> str:
    """Hash das condicoes sob as quais a linha foi produzida.

    Correcao B-10. O que torna duas linhas comparaveis nao e o rotulo da montagem,
    e o conjunto de condicoes. Sem carimbo, uma run feita antes de uma correcao e
    outra feita depois concatenam sem erro e produzem uma tabela cuja diferenca
    entre colunas mistura o efeito medido com a mudanca de pipeline — que e
    exatamente como as grades invalidadas chegaram ate aqui.
    """
    d = {k: cfg[k] for k in CAMPOS_CARIMBO if k in cfg}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def banda_efetiva(X: np.ndarray, sfreq: float, frac: float = 0.98):
    """Banda que de fato contem `frac` da potencia do sinal ENTREGUE ao modelo.

    Propriedade medida, nao declarada: e o que teria pego B-1, onde a banda
    declarada era 4 a 40 e a aplicada era 8 a 30. Nenhum portao de declaracao pegaria
    aquilo, porque nenhuma string estava errada.
    """
    n = min(len(X), 32)
    nper = int(min(256, X.shape[-1]))
    f, p = signal.welch(X[:n], fs=sfreq, nperseg=nper, axis=-1)
    p = p.mean(axis=(0, 1))
    c = np.cumsum(p)
    if c[-1] <= 0:
        return float("nan"), float("nan")
    c = c / c[-1]
    m = (1.0 - frac) / 2.0
    return float(f[np.searchsorted(c, m)]), float(f[min(np.searchsorted(c, 1 - m), len(f) - 1)])


def medidas(X: np.ndarray, sfreq: float, modelo=None) -> dict:
    """As propriedades medidas que acompanham o carimbo em cada linha do CSV.

    `n_ch_efetivo` e o que o harness ENTREGA; `n_ch_modelo` e o que o modelo declara
    usar. Os dois divergem quando a arquitetura descarta canais por conta propria — o
    EEGSym descarta laterais sem par (correcao D-5) — e e essa divergencia que uma
    curva de desempenho por densidade precisa enxergar para nao plotar o ponto na
    abscissa errada. `n_comp_efetivo` faz o mesmo para a capacidade dos metodos CSP,
    que sob CAR fica limitada ao posto (correcoes D-3 e D-4); 0 significa que o
    modelo nao tem esse conceito.
    """
    lo, hi = banda_efetiva(X, sfreq)
    return {"n_ch_efetivo": int(X.shape[1]),
            "n_ch_modelo": int(getattr(modelo, "n_canais_efetivo", X.shape[1])),
            "n_comp_efetivo": int(getattr(modelo, "n_components_efetivo", 0)),
            "sd_entrada_uv": float(np.std(X)),
            "banda_ef_lo": lo,
            "banda_ef_hi": hi}


# Carimbos de definicoes ANTERIORES do proprio carimbo, com condicoes identicas.
# Acrescentar um campo a CAMPOS_CARIMBO muda o hash de execucoes cujas condicoes
# nao mudaram; sem este registro a grade ja concluida pareceria incomparavel com
# uma reexecucao identica. Cada entrada precisa dizer o que mudou e por que a
# equivalencia vale.
EQUIVALENTES = {
    # `fases_treino` entrou no carimbo quando o braco de fase unica foi criado.
    # A grade de 7920 celulas concluida em 11/08 rodou com "real+mi", que e o
    # default, entao as condicoes sao as mesmas do carimbo novo.
    "1bcd8582ef": "854a2f338d",
}


def recusa_mistura(carimbos) -> None:
    """Aborta se linhas de contratos diferentes forem juntadas na mesma tabela."""
    u = sorted({EQUIVALENTES.get(c, c) for c in carimbos if c})
    if len(u) > 1:
        raise ValueError(
            "linhas de contratos diferentes na mesma tabela: " + ", ".join(u) +
            ". Compare-as como experimentos distintos ou refaca sob um so contrato.")


# ---------------------------------------------------------------------------
#  6. Construcao dos modelos sob o contrato (B-2, B-4, B-6)
# ---------------------------------------------------------------------------

def _so_aceitos(fn, kw: dict) -> dict:
    """Filtra kwargs pela assinatura da fabrica. O que nao passar e verificado
    depois na instancia, por `verifica_instancia` — declarar nao basta."""
    p = inspect.signature(fn).parameters
    if any(q.kind == inspect.Parameter.VAR_KEYWORD for q in p.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in p}


def constroi(nome: str, cfg: dict):
    """Instancia o modelo sob o contrato. `filter_band=None` em todos, sem excecao."""
    C = dict(epochs=cfg["teto_epocas"], patience=cfg["paciencia"],
             val_split=cfg["val_split"], optimizer=cfg["otimizador"],
             weight_decay=cfg["weight_decay"], lr=cfg["lr"],
             batch_size=cfg["batch_size"], filter_band=None, sfreq=cfg["sfreq"])
    ft = dict(epochs=cfg["calib_epocas"], lr=cfg["calib_lr"],
              patience=cfg["calib_paciencia"])

    F2 = dict(epochs=cfg["teto_epocas"], lr=cfg["lr"], patience=cfg["paciencia"])

    def _liga(model, metodo, **kw):
        """Anexa um metodo pre-configurado que SOBREVIVE a `copy.deepcopy`.

        Correcao B-A, achada no pre-voo. A versao anterior fazia:

            orig = model.continue_training      # bound method do objeto ORIGINAL
            def _wrap(X, y): return orig(X, y, **ft)
            model.continue_training = _wrap     # atributo de INSTANCIA

        `copy.deepcopy` trata funcoes como atomicas, entao a copia recebia o MESMO
        objeto-funcao, cuja celula de closure continuava apontando para o modelo
        original. No laco do runner (`c = deepcopy(clf); _calibra(c, ...)`) a
        calibracao treinava `clf`, o modelo do pool, e `c.predict` avaliava um modelo
        que nunca foi calibrado. Consequencias medidas nos cinco profundos: no fold 0
        o delta de pesos da copia era exatamente 0.0, e nos folds seguintes o `clf`
        ja carregava as epocas de calibracao dos folds anteriores, das quais 82% a 98%
        pertenciam ao teste do fold corrente. Assinatura na grade antiga, que tem o
        mesmo defeito herdado de `run_all_models.py`: acuracia crescendo monotonica
        com o INDICE DO FOLD, que sob StratifiedKFold de semente fixa e apenas ordem
        de execucao — eegsym +0,201 (p=3e-60), eegnet +0,168, shallow +0,109, contra
        +0,005 / +0,003 / +0,013 nos tres modelos que nao passavam pelo wrapper.

        `functools.partial` resolve porque o deepcopy o RECONSTROI, e o `memo` ja
        contem model -> copia quando os argumentos sao copiados. O redespacho por
        `type(self)` garante que a funcao chamada seja a da classe, nunca um bound
        method congelado.
        """
        def _chama(self, X, y):
            return getattr(type(self), metodo)(self, X, y, **kw)
        return functools.partial(_chama, model)

    def _com_regimes(model):
        """Dois regimes DISTINTOS e explicitos, em vez de um wrapper para os dois.

        Correcao D-1. Antes, o mesmo `_wrap` servia a Fase 2 do pool e a calibracao
        do alvo, entao os profundos treinavam a Fase 2 com 80 epocas e lr 1e-4
        enquanto o carimbo gravava as 200 epocas e lr 1e-3 do contrato — declarado e
        executado divergiam na fase mais cara do pipeline. Nem passar
        `continue_training(X, y)` sem argumentos resolveria: os cinco modelos trazem
        `epochs: int = 80` fixo na assinatura, e so `lr` e `patience` caem em
        `self.*`. O regime do contrato precisa ser passado, nao herdado.
        """
        model.fase2_pool = _liga(model, "continue_training", **F2)
        model.calibra_alvo = _liga(model, "continue_training", **ft)
        return model

    _com_calibracao = _com_regimes   # nome antigo, usado abaixo

    if nome == "csp_lda":
        from csp_lda.csp_lda import create_csp_lda_setup1_v1_model
        return create_csp_lda_setup1_v1_model(n_components=6)

    if nome == "fbcsp_svm":
        from fbcsp_svm.fbcsp_svm import create_fbcsp_svm_setup1_v1_model
        return create_fbcsp_svm_setup1_v1_model(
            n_csp_components=4, n_features=12, svm_C=1.0, svm_kernel="linear")

    if nome == "riemannian":
        from riemannian_dcca.riemannian_dcca import create_riemannian_setup1_v1_model
        # B-2: quem seleciona canais e o harness. Explicito mesmo com o default ja
        # invertido, porque foi um default silencioso que produziu o defeito.
        return create_riemannian_setup1_v1_model(
            use_channel_subset=False, preprocessed_externally=True)

    if nome == "eegnet":
        from eegnet.eegnet import create_eegnet_setup1_v7_model as f
        return _com_calibracao(f(**_so_aceitos(f, dict(C, F1=16, D=4, F2=64, dropout=0.5))))

    if nome == "eegsym":
        from eegsym.eegsym import create_eegsym_setup1_model as f
        return _com_calibracao(f(**_so_aceitos(f, dict(C, n_filters=8, dropout=0.4))))

    if nome == "shallow_convnet":
        from shallow_convnet.shallow_convnet import create_shallow_convnet_setup1_model as f
        return _com_calibracao(f(**_so_aceitos(f, dict(
            C, n_filters=40, kernel_temporal=25, pool_size=75, pool_stride=15,
            dropout=0.5))))

    if nome == "deepcnn":
        from cnn.adaptive_cnn import create_deepcnn_setup1_model as f
        # optimizer/weight_decay vem do contrato (adam/1e-4) e nao do default da
        # classe (adamw/1e-3), que nao tinha fonte e divergia dos outros quatro.
        return _com_calibracao(f(**_so_aceitos(f, dict(C, dropout=0.5))))

    if nome == "stia_net":
        from stia_net.stia_net import create_stianet_setup1_model as f
        return _com_calibracao(f(**_so_aceitos(f, dict(
            C, gcn_hidden=64, gcn_out=32, tcn_filters=(16, 32, 64), n_heads=8,
            dropout=0.5))))

    raise KeyError(f"modelo sem receita no contrato: {nome}")


# ---------------------------------------------------------------------------
#  7. Portao 1: declaracao (secao 4.3)
# ---------------------------------------------------------------------------

_ATRIBUTOS = {"filter_band": None, "val_split": "val_split", "epochs": "teto_epocas",
              "patience": "paciencia", "optimizer": "otimizador",
              "weight_decay": "weight_decay", "lr": "lr", "batch_size": "batch_size"}


def verifica_instancia(nome: str, model, cfg: dict) -> list[str]:
    """Le os atributos da instancia e confronta com o contrato.

    Verificacao comportamental, nao textual: se o modelo guardou outro valor, o
    desacordo aparece aqui mesmo que a declaracao esteja perfeita. Atributo ausente
    nao e falha — nem todo modelo parametriza tudo (o EEGSym fixa o otimizador no
    codigo) — mas atributo presente e divergente e.
    """
    ruins = []
    for attr, chave in _ATRIBUTOS.items():
        if not hasattr(model, attr):
            continue
        obtido = getattr(model, attr)
        esperado = None if chave is None else cfg[chave]
        if attr == "optimizer" and isinstance(obtido, str):
            obtido = obtido.lower()
        if obtido != esperado:
            ruins.append(f"{nome}.{attr}={obtido!r} (contrato: {esperado!r})")
    # Correcao D-8: a uniao das sub-bandas de um banco de filtros TEM de igualar a
    # banda do contrato (secao 4.2). Sem esta checagem, construir o fbcsp com
    # `freq_bands=[(8,30)]` produzia uma celula de banda estreita rodando sob carimbo
    # de 4-40: `portao_declaracao` nao pegava, porque `banco_de_bandas` e divergencia
    # permitida e nao esta em UNIFORMES_PROIBIDOS, e a instancia nunca era lida.
    bandas = getattr(model, "freq_bands", None)
    if bandas:
        lo, hi = min(b[0] for b in bandas), max(b[1] for b in bandas)
        alvo = cfg["banda_hz"]
        if abs(lo - alvo[0]) > 1e-6 or abs(hi - alvo[1]) > 1e-6:
            ruins.append(f"{nome}.freq_bands cobre {lo}-{hi} Hz; o contrato exige "
                         f"{alvo[0]}-{alvo[1]} Hz na uniao das sub-bandas")
    if not (hasattr(model, "predict_proba") or hasattr(model, "predict_proba_three_class")):
        ruins.append(f"{nome}: sem saida macia — inadmissivel na tabela primaria")
    return ruins


def portao_declaracao(nomes, cfg: dict) -> None:
    """Roda antes da primeira epoca. Falha por omissao: modelo sem entrada aborta."""
    erros = []
    for n in nomes:
        e = MODELOS.get(n)
        if e is None:
            erros.append(f"{n}: sem entrada no contrato")
            continue
        for k, d in e["divergencias"].items():
            faltando = [c for c in ("valor", "motivo", "fonte", "exige_gemea")
                        if d.get(c) in (None, "")]
            if faltando:
                erros.append(f"{n}.{k}: divergencia sem {', '.join(faltando)}")
            if k in UNIFORMES_PROIBIDOS:
                erros.append(f"{n}.{k}: e propriedade uniforme, nao pode divergir")
    if erros:
        raise SystemExit("PORTAO 1 (declaracao) reprovou:\n  " + "\n  ".join(erros))


def gemeas_pendentes(nomes) -> list[str]:
    """Modelos cuja leitura fiel ao artigo exige uma celula gemea de banda."""
    return [f"{n}.{k}" for n in nomes
            for k, d in MODELOS.get(n, {}).get("divergencias", {}).items()
            if d["exige_gemea"]]
