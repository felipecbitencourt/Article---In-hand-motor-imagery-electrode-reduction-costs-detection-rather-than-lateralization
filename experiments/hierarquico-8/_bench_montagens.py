# -*- coding: utf-8 -*-
"""Benchmark por montagem: N modelos x M montagens, com salvamento por sujeito.

DESENHO

Escada aninhada 2 - 4 - 8 - 16 - 21 - 32 - 64, derivada por proximidade ao
cortex motor da mao (distancia minima a C3/C4 no arranjo standard_1005),
preservando intactas as montagens de 8 e 21 do artigo. O aninhamento e a
propriedade que permite atribuir a diferenca a densidade, e nao a regiao
amostrada; ele e verificado na partida e o script aborta se falhar.

Montagens de controle, que respondem "quais" em vez de "quantos":
  sem_fz      anat8 com Fz -> C1. O Fz e o mais contaminado por EOG dos oito, e
              como o CAR e calculado sobre os canais disponiveis, num conjunto de
              8 seus artefatos se propagam a todos os outros — diluicao que
              existe em 64 canais e nao existe em 8.
  posterior8  P3,P4,PO3,PO4,O1,O2,P7,P8. O ranking CSP elegeu 7 de 8 canais
              posteriores do hemisferio direito, com frontopolares logo atras —
              padrao de OLHAR para o alvo lateralizado, nao de imageamento
              motor, que seria bilateral em torno de C3/C4. Se `posterior8`
              bater `anat8`, os modelos exploram informacao lateralizada nao
              motora e parte da penalidade 64->8 e perda de confundidor.
  ocular4     Fp1,Fp2,AF7,AF8. Se estes sozinhos decodificam, e movimento ocular.

PRE-PROCESSAMENTO: DONO UNICO

O pipeline do artigo rejeita epocas no sinal BRUTO de 64 canais ANTES de
selecionar a montagem. Duas consequencias, ambas auditadas:
  - a montagem reduzida herda um filtro de qualidade derivado de eletrodos que
    nunca usa (25 de ~45 epocas mudam de status se calculado sobre os 8 em uso);
  - o limiar fixo de 100 uV dispara a trava de 50% em 26 de 30 alvos, e o
    criterio efetivo vira "manter as 50% de menor amplitude".

Aqui a ordem e: seleciona canais -> CAR -> filtra 4-40 Hz -> rejeita por IQR ->
converte para microvolts, tudo em `_contrato.prepara`, uma vez so, e o resultado
e o que o modelo recebe. Nenhum modelo filtra, referencia, recorta canais ou
reescala de novo, e cada um e CONSTRUIDO de forma a nao poder faze-lo.

A versao intermediaria deste arquivo entregava o sinal CRU e deixava cada modelo
aplicar a propria banda. Isso parecia evitar filtragem dupla, mas colocava a banda
efetiva fora do controle do harness — e foi assim que a rejeicao passou meses
decidindo em 8-30 Hz enquanto o cabecalho declarava 4-40. Ver `_contrato.py`.

SALVAMENTO

Uma linha de metricas e as predicoes por epoca sao gravadas e descarregadas ao
fim de CADA fold; o estado e reconstruido de (montagem, modelo, alvo, fold) na
retomada. Uma queda perde no maximo o fold em curso.

Uso:
  python _bench_montagens.py --montagens 8 21 --modelos eegnet csp_lda
  python _bench_montagens.py --montagens todas --modelos acima_do_acaso
"""
from __future__ import annotations

import sys, csv as _csv, copy, time, argparse, warnings, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve(); EXP = HERE.parent; REPO = EXP.parent.parent
for p in (str(REPO), str(REPO / "modelos"), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

import torch
from sklearn.model_selection import StratifiedKFold

from core.data_loader import load_subject_setup1_hand_lateral, load_run
from core.metrics import compute_metrics_setup1_hierarchical
import _contrato as CT

# Sujeitos excluidos do POOL de treino. Vem de run_setup1_base_iqr.py, onde foram
# estabelecidos por auditoria CSP+LDA intra-sujeito: sao registros cuja qualidade
# de imageamento nao se distingue do acaso, e que portanto injetam rotulos
# ruidosos no pool. Nenhum deles esta entre os 30 alvos, entao a exclusao afeta so
# o treino; o conjunto de teste permanece intacto. Pool: 108 -> 102 sujeitos.
BAD_SUBJECTS = frozenset({38, 88, 89, 92, 100, 104})

# Escada derivada por proximidade a C3/C4; 8 e 21 sao os do artigo.
#
# SIMETRIA HEMISFERICA. A proximidade pura a C3/C4 nao garante pares esquerda /
# direita, e a versao anterior do 16 saiu com 8 canais a esquerda contra 6 a
# direita. Duas consequencias: o EEGSym, cuja arquitetura divide a entrada em dois
# ramos de tamanho igual, nao consegue processar a montagem; e um revisor pode
# alegar que a lateralidade do Nivel 2 e favorecida de um lado.
#
# A segunda alegacao foi medida e nao se sustenta — a diferenca de sensibilidade
# entre as duas maos e de +0,8 pp no 16 e +3,0 pp no 21 (p = 0,72 e 0,27), e a
# montagem de 2 canais, perfeitamente simetrica, exibe desvio maior (-4,1 pp).
# O que se ve e dispersao entre sujeitos, nao efeito da montagem.
#
# Ainda assim o 16 foi refeito simetrico: 6 a esquerda, 6 a direita, 4 na linha
# media. Ele e novo, nao tem compromisso com a Tabela 1 do artigo, continua
# contendo o 8 e contido no 21, e assim destrava o EEGSym nesta densidade.
#
# O 21 permanece assimetrico (9L / 8R, CP5 sem par CP6) porque e o cinturao
# anatomico publicado e mexer nele quebraria a comparabilidade que justifica a
# escada existir. A assimetria e propriedade do cinturao, nao escolha nossa, e
# esta declarada como limitacao.
#
# Nao se usou o ranking CSP para definir a escada, e a tentacao de justifica-la
# assim a posteriori foi descartada: o ranking elege CP4, P4, P6, CP6, C4, PO4
# como primeiros colocados, sem nenhum canal do hemisferio esquerdo entre os nove
# primeiros. Uma selecao orientada a dados iria para occipital DIREITO — o padrao
# que atribuimos a olhar para o alvo, e que motivou a posterior8 existir. Alem de
# circular (selecionar canais nos mesmos dados em que se avalia e vazamento de
# selecao), a justificativa pediria a assimetria inversa da que temos.
MONTAGENS = {
    "2":  ['C3', 'C4'],
    "4":  ['C3', 'C4', 'CP3', 'CP4'],
    "8":  ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'Fz'],
    "16": ['C3','C4','Cz','FC3','FC4','CP3','CP4','Fz',
           'C1','C2','C5','C6','FC1','FC2','FCz','CPz'],
    "21": ['C3','C4','Cz','C1','C2','C5','C6','FC1','FC2','FC3','FC4','FC5','FC6','FCz',
           'CP1','CP2','CP3','CP4','CP5','CPz','Fz'],
    "32": ['C3','C4','Cz','C1','C2','C5','C6','FC1','FC2','FC3','FC4','FC5','FC6','FCz',
           'CP1','CP2','CP3','CP4','CP5','CPz','Fz',
           'CP6','F3','F4','F5','F6','P3','P4','P5','P6','T7','T8'],
    "64": None,   # todos
    "3":       ['C3', 'Cz', 'C4'],          # ancora bibliografica classica
    "sem_fz":  ['C3','C4','Cz','FC3','FC4','CP3','CP4','C1'],
    "posterior8": ['P3','P4','PO3','PO4','O1','O2','P7','P8'],
    "ocular4":    ['Fp1','Fp2','AF7','AF8'],
}
ESCADA = ["64", "32", "21", "16", "8", "4", "2"]
ACIMA_DO_ACASO = ["eegnet", "shallow_convnet", "riemannian", "fbcsp_svm", "csp_lda"]

MET_COLS = None   # definido na primeira metrica, para gravar as 30


def nomes_64() -> list[str]:
    import mne
    raw = load_run(1, 4); mne.datasets.eegbci.standardize(raw)
    return raw.info["ch_names"]


def valida_aninhamento(todos):
    """A escada precisa ser aninhada; sem isso a comparacao mede regiao, nao densidade."""
    for a, b in zip(ESCADA[::-1], ESCADA[::-1][1:]):
        ca = set(MONTAGENS[a] or todos); cb = set(MONTAGENS[b] or todos)
        if not ca.issubset(cb):
            raise SystemExit(f"ABORTA: montagem {a} nao esta contida em {b} — "
                             f"faltam {sorted(ca - cb)}")
    print("  aninhamento 2<4<8<16<21<32<64 verificado")


# Montagens que se sabe assimetricas e que assim permanecem por decisao explicita.
# Qualquer OUTRA que aparecer assimetrica e acidente e aborta a run.
#   21      cinturao anatomico publicado (9E/8D, CP5 sem par CP6). Mexer quebraria
#           a comparabilidade com a Tabela 1, que e o motivo de a escada existir.
#   sem_fz  a substituicao Fz -> C1 desbalanceou (4E/3D). Ja esta completa nos 8
#           modelos e seu resultado e um achado central (empata com posterior8);
#           refaze-la custaria 8 combinacoes para corrigir um vies de -0,5 pp
#           (p = 0,81). Fica declarado que o contraste com anat8 confunde
#           "sem Fz" com "assimetrica".
ASSIMETRICAS_ACEITAS = {"21", "sem_fz"}


def lado(ch: str) -> str:
    """Hemisferio pelo sufixo numerico da nomenclatura 10-05: impar E, par D, z centro."""
    d = "".join(c for c in ch if c.isdigit())
    if not d:
        return "C" if ch.lower().endswith("z") else "?"
    return "E" if int(d) % 2 else "D"


def valida_simetria(todos):
    """Relata o balanco esquerda/direita e aborta se surgir assimetria nao prevista.

    Importa por dois motivos: o EEGSym exige ramos hemisfericos de tamanho igual, e
    uma montagem desbalanceada e alvo facil de revisor no Nivel 2, que e justamente
    a discriminacao esquerda/direita.
    """
    print(f"  {'montagem':<12}{'E':>3}{'D':>3}{'centro':>7}  simetria")
    for mt in ESCADA + ["3", "sem_fz", "posterior8", "ocular4"]:
        chs = MONTAGENS[mt] or todos
        e = sum(1 for c in chs if lado(c) == "E")
        d = sum(1 for c in chs if lado(c) == "D")
        c = sum(1 for c in chs if lado(c) == "C")
        ok = e == d
        if not ok and mt not in ASSIMETRICAS_ACEITAS:
            raise SystemExit(f"ABORTA: montagem {mt} assimetrica ({e}E/{d}D) e nao "
                             f"consta em ASSIMETRICAS_ACEITAS. Se for intencional, "
                             f"declare la e justifique.")
        nota = "sim" if ok else "NAO (aceita por decisao)"
        print(f"  {mt:<12}{e:>3}{d:>3}{c:>7}  {nota}")


def desliga_car():
    """Substitui o CAR pela identidade em todos os modelos, para medir o confundidor.

    O CAR subtrai a media dos canais DISPONIVEIS, entao remove sempre exatamente um
    grau de liberdade — a media passa a ser zero por construcao. A fracao perdida
    porem cresce conforme a montagem encolhe: 1 de 64 e 1,6%, mas 1 de 2 e 50%.
    Medido no sujeito 1, o posto pos-CAR e 63, 15, 7, 3, 2 e 1 para 64, 16, 8, 4, 3
    e 2 canais; em 2 canais a covariancia fica com autovalores separados por 16
    ordens de grandeza, ou seja singular, e metodos baseados em covariancia (CSP,
    FBCSP, Riemannian) passam a operar sobre a regularizacao em vez do sinal.

    Isso e um confundidor do proprio eixo que o benchmark mede: parte da penalidade
    atribuida a "menos eletrodos" pode ser o CAR consumindo uma fatia
    proporcionalmente maior do subespaco. Rodar a escada sem CAR separa as duas
    coisas.

    O patch e feito ANTES do primeiro import de modelo: cada um faz `from
    core.preprocessing import common_average_reference`, e essa ligacao le o atributo
    do modulo no momento do import. Patchear depois nao teria efeito.

    Correcao B-9: o braco tambem passa `car=False` a `_contrato.prepara`, e la a
    DECISAO de rejeicao continua sendo tomada com CAR. Sem isso o braco mudaria
    tambem quais epocas sobrevivem, e a diferenca medida entre os dois bracos
    misturaria o efeito da referencia com o efeito da retencao — mediria duas coisas
    e reportaria uma.
    """
    CT.desliga_car_nos_modelos()
    print("  CAR DESLIGADO — controle do confundidor de posto "
          "(a retencao de epocas nao muda: a decisao segue com CAR)")


def _fase1(model, X, y):
    """Mesmo despacho de run_setup1_base._train_phases: nem todo modelo expoe .fit."""
    if hasattr(model, "fit_phase1_real"): model.fit_phase1_real(X, y)
    elif hasattr(model, "fit"): model.fit(X, y)
    else: raise TypeError(f"{type(model).__name__} sem fit_phase1_real nem fit")


def _fase2(model, X, y):
    # `fase2_pool` vem de _contrato._com_regimes e carrega o regime do CONTRATO
    # (200 epocas, lr 1e-3, paciencia 20). Tem de vir antes de `continue_training`,
    # cuja assinatura traz `epochs=80` fixo — ver a correcao D-1.
    if hasattr(model, "fase2_pool"): model.fase2_pool(X, y)
    elif hasattr(model, "fit_phase2_mi"): model.fit_phase2_mi(X, y)
    elif hasattr(model, "continue_training"): model.continue_training(X, y)
    else: raise TypeError(f"{type(model).__name__} sem fit_phase2_mi nem continue_training")


def _calibra(model, X, y):
    # `calibra_alvo` carrega o regime de calibracao (80 epocas, lr 1e-4, paciencia
    # 15) e, ao contrario do wrapper que substituiu, sobrevive ao copy.deepcopy do
    # laco de folds — ver a correcao B-A em _contrato._liga.
    if hasattr(model, "calibra_alvo"): model.calibra_alvo(X, y)
    elif hasattr(model, "fine_tune_subject"): model.fine_tune_subject(X, y)
    elif hasattr(model, "fit_phase2_mi"): model.fit_phase2_mi(X, y)
    else: raise TypeError(f"{type(model).__name__} sem metodo de calibracao")


def grupos_de_alvos(targets, k, seed):
    """Divide os alvos em k grupos por sorteio com semente fixa.

    Cada grupo e excluido do pool que treina o modelo que o avalia, entao a
    garantia essencial se mantem: nenhum alvo participa do treino do modelo que o
    testa. Nao e holdout — ha rotacao, e todo alvo continua sendo testado.

    O que muda em relacao a leave-one-out e que os alvos de um mesmo grupo
    COMPARTILHAM o modelo, o que correlaciona seus erros e enfraquece a premissa de
    independencia dos testes pareados. Por isso a coluna `grupo` vai para o CSV: a
    analise precisa poder levar isso em conta.

    Medido antes de adotar: com pools de 102, 97 e 93 sujeitos, o maior efeito sobre
    a acuracia foi de 0,08 pp, sem direcao monotonica — quarenta vezes menor que a
    variacao de ~3 pp da propria pipeline entre execucoes. O tamanho do pool esta
    saturado nessa faixa, entao K se escolhe pelo argumento de independencia.
    """
    rng = np.random.default_rng(seed)
    ordem = list(targets); rng.shuffle(ordem)
    return [sorted(ordem[i::k]) for i in range(k)]


def carrega_pool(pool, fase, idx, cfg):
    """Carrega o pool tolerando falha de sujeito individual (correcao D-9).

    Antes, `carrega` do pool rodava numa list comprehension dentro do try do grupo:
    um unico sujeito ruim entre os ~98 levantava, e o `except` descartava o GRUPO
    INTEIRO — ate 15 celulas (5 alvos x 3 folds) — imprimindo uma linha de erro no
    meio de milhares e terminando com `BENCH_MONTAGENS_DONE`. Como nenhuma linha era
    gravada, a retomada refazia o mesmo caminho e o buraco era permanente se a causa
    fosse deterministica.

    Um sujeito a menos num pool de ~98 nao muda nada de material; um grupo a menos
    muda. Por isso o pool tolera e REGISTRA, enquanto o alvo continua sendo condicao
    de erro: no alvo, a epoca perdida e a medida.
    """
    Xs, ys, ruins = [], [], []
    for s_ in pool:
        try:
            X, y, _ = carrega(s_, fase, idx, cfg)
            if len(X): Xs.append(X); ys.append(y)
        except CT.RejeicaoExcessiva:
            ruins.append(s_)
    if not Xs:
        raise CT.RejeicaoExcessiva(f"pool inteiro fora na fase {fase}")
    return np.concatenate(Xs), np.concatenate(ys), ruins


def carrega(sujeito, fase, idx, cfg):
    """Carrega um sujeito e uma fase sob o contrato. Devolve `(X_uv, y, keep)`.

    Toda a decisao de pre-processamento esta em `_contrato.prepara`; aqui so entra o
    que e especifico do dataset. `keep` sao os indices das epocas mantidas dentro do
    array original do sujeito, e e o identificador estavel exigido por B-3.

    A rejeicao acontece POR SUJEITO e POR FASE (B-7): esta funcao nunca ve o pool
    concatenado, entao o limiar de um sujeito nunca e calculado sobre a distribuicao
    de amplitudes de outro.
    """
    X, y = load_subject_setup1_hand_lateral(sujeito, fase, tmin=cfg["janela_s"][0],
                                            tmax=cfg["janela_s"][1])
    if len(X) == 0:
        return X, y, np.array([], dtype=int)
    Xp, keep = CT.prepara(X, idx, car=cfg["car"], sfreq=cfg["sfreq"],
                          banda=cfg["banda_hz"], ordem=cfg["ordem_filtro"],
                          iqr_k=cfg["iqr_k"], trava=cfg["trava_min_keep"],
                          rotulo=f"S{sujeito:03d}/{fase}")
    return Xp, y[keep], keep


def _sara_cauda(f: Path) -> int:
    """Descarta uma ultima linha truncada, e devolve quantos bytes cortou.

    Correcao D-12. Uma queda nao graciosa no meio de um `writerow` deixa meia
    linha no arquivo; na retomada o `append` gruda a proxima linha nela e o
    resultado e um registro soldado que o pandas nao le — ou, pior, le errado.
    Como toda linha completa termina em quebra, basta truncar ate a ultima.
    """
    if not f.exists() or f.stat().st_size == 0:
        return 0
    with open(f, "rb+") as h:
        h.seek(0, 2); fim = h.tell()
        h.seek(max(0, fim - 65536)); cauda = h.read()
        QUEBRA = b"\n"
        if cauda.endswith(QUEBRA):
            return 0
        corte = cauda.rfind(QUEBRA)
        if corte < 0:
            return 0
        novo = fim - (len(cauda) - corte - 1)
        h.truncate(novo)
        return fim - novo


def main():
    global MET_COLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--montagens", nargs="+", default=["8"])
    ap.add_argument("--modelos", nargs="+", default=["eegnet"])
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[105,106,107,108,109,3,7,12,18,22,25,31,34,41,45,
                             49,52,57,60,63,68,71,76,79,83,87,91,95,98,102])
    ap.add_argument("--subjects", type=int, nargs=2, default=(1, 109))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=2.5)
    ap.add_argument("--grupos", type=int, default=6,
                    help="numero de grupos de alvos; um treino de pool por grupo em "
                         "vez de um por alvo. 6 grupos de 5 alvos: pool de 98, ganho "
                         "de 5x, 6 modelos independentes. Use --grupos 30 para "
                         "leave-one-out.")
    ap.add_argument("--pool-max", type=int, default=None,
                    help="subamostra o pool para N sujeitos (semente fixa). Isola o "
                         "efeito do TAMANHO do pool, sem implementar o agrupamento — "
                         "e o que decide quantos grupos sao defensaveis.")
    ap.add_argument("--out-dir", default=str(EXP / "results" / "bench_montagens"))
    ap.add_argument("--sem-car", action="store_true",
                    help="desliga o CAR em todos os modelos; ver desliga_car()")
    ap.add_argument("--banda", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="banda do contrato em Hz (padrao 4 40). Use --banda 8 30 "
                         "para a celula GEMEA de riemannian e stia_net, cuja leitura "
                         "fiel ao artigo de base pede banda estreita. Muda o carimbo, "
                         "entao as duas nao entram na mesma tabela por acidente.")
    ap.add_argument("--ordem-filtro", type=int, default=None,
                    help="ordem do Butterworth (padrao 5)")
    ap.add_argument("--fases", choices=("real+mi", "mi"), default=None,
                    help="fases de treino do pool. 'real+mi' (padrao) treina em "
                         "movimento executado e depois em imageamento; 'mi' treina "
                         "uma vez so, sobre imageamento, e mede quanto a fase de "
                         "movimento real contribui. Muda o carimbo.")
    ap.add_argument("--calib-n", type=int, default=None,
                    help="orcamento ABSOLUTO de epocas de calibracao por fold "
                         "(padrao 48). 0 desliga o orcamento e usa todo o lado de "
                         "treino do fold, que e o comportamento antigo — sob ele a "
                         "retencao e a calibracao variam juntas com a densidade.")
    A = ap.parse_args()

    # Contrato desta run: o default e CT.CONTRATO, e cada desvio precisa ser
    # pedido explicitamente na linha de comando. O carimbo resume tudo (B-10).
    cfg = dict(CT.CONTRATO)
    cfg["janela_s"] = (A.tmin, A.tmax)
    cfg["folds"] = A.folds
    cfg["car"] = not A.sem_car
    cfg["seed"] = A.seed
    cfg["grupos"] = A.grupos
    cfg["pool_max"] = A.pool_max or 0
    if A.banda is not None:
        cfg["banda_hz"] = (float(A.banda[0]), float(A.banda[1]))
    if A.ordem_filtro is not None:
        cfg["ordem_filtro"] = A.ordem_filtro
    if A.calib_n is not None:
        cfg["calib_n"] = None if A.calib_n <= 0 else A.calib_n
    if A.fases is not None:
        cfg["fases_treino"] = A.fases
    SELO = CT.carimbo(cfg)

    mts = ESCADA if A.montagens == ["todas"] else A.montagens
    mds = ACIMA_DO_ACASO if A.modelos == ["acima_do_acaso"] else A.modelos

    todos = nomes_64()
    print(f"CUDA={torch.cuda.is_available()} | montagens={mts} | modelos={mds} "
          f"| alvos={len(A.targets)} | folds={A.folds}")
    print(f"  contrato {SELO} | banda {cfg['banda_hz'][0]:.0f}-{cfg['banda_hz'][1]:.0f} Hz "
          f"ordem {cfg['ordem_filtro']} | CAR {'on' if cfg['car'] else 'OFF'} | "
          f"IQR k={cfg['iqr_k']} | calib {cfg['calib_n']} | fases {cfg['fases_treino']}")
    valida_aninhamento(todos)
    valida_simetria(todos)

    # Portao 1 (secao 4.3): falha por omissao, antes de qualquer epoca ser lida.
    CT.portao_declaracao(mds, cfg)
    pend = CT.gemeas_pendentes(mds)
    if pend and cfg["banda_hz"] == CT.CONTRATO["banda_hz"]:
        print(f"  celulas gemeas pendentes: {', '.join(pend)} "
              f"(rodar de novo com --banda 8 30)")

    grupos = grupos_de_alvos(A.targets, A.grupos, A.seed)
    print(f"  {len(grupos)} grupos de alvos (semente {A.seed}), "
          f"pool = {109 - len(BAD_SUBJECTS) - len(grupos[0])} sujeitos:")
    for i, g in enumerate(grupos, 1):
        print(f"    grupo {i}: {g}")

    if A.sem_car:
        desliga_car()

    out = Path(A.out_dir); out.mkdir(parents=True, exist_ok=True)
    fmet, fpre = out / "metricas.csv", out / "predicoes.csv"

    # A cauda e reparada ANTES da leitura de retomada, e a ordem importa: medida na
    # verificacao, repara-la depois marcava a celula meio-gravada como concluida e em
    # seguida apagava a linha dela, perdendo a celula em silencio — soma(n_teste) e
    # len(predicoes) divergiam e nada falhava.
    for _f in (fmet, fpre):
        _n = _sara_cauda(_f)
        if _n:
            print(f"  linha truncada descartada em {_f.name} ({_n} bytes) — "
                  f"queda anterior no meio da gravacao")

    # Retomada SO dentro do mesmo contrato (B-10). Anexar a um CSV produzido sob
    # outras condicoes e como as grades invalidadas se formaram: nada falha, e a
    # tabela final compara colunas que nao sao comparaveis.
    feito, cab_existente = set(), []
    if fmet.exists() and fmet.stat().st_size > 0:
        _leitor = _csv.DictReader(open(fmet, encoding="utf-8"))
        linhas = list(_leitor)
        cab_existente = list(_leitor.fieldnames or [])
        antigos = {r.get("carimbo") for r in linhas}
        if antigos - {SELO}:
            raise SystemExit(
                f"ABORTA: {fmet} tem linhas do(s) contrato(s) {sorted(antigos)} e esta "
                f"run e {SELO}. Use --out-dir novo; nao ha retomada entre contratos.")
        for r in linhas:
            feito.add((r["montagem"], r["modelo"], int(r["target"]), int(r["fold"])))
        print(f"  retomando: {len(feito)} celulas ja concluidas sob {SELO}")

        # RECONCILIACAO BIDIRECIONAL entre metricas.csv e predicoes.csv.
        #
        # A linha de metricas e o marcador de commit, entao o estado consistente e:
        # toda celula com marcador tem EXATAMENTE n_teste predicoes. Qualquer desvio
        # — predicao orfa de celula sem marcador, ou celula com predicoes de menos —
        # e resto de queda, e a celula inteira e desfeita para ser refeita limpa.
        #
        # Medido antes disto: uma queda no meio de um fold deixava 27 predicoes
        # duplicadas na celula refeita, ou uma predicao a menos numa celula que
        # constava como completa. Nenhum dos dois falhava em lugar nenhum, e
        # `predicoes.csv` e a base da matriz de decisao e da analise de
        # ortogonalidade — a razao de B-3 existir.
        if fpre.exists() and fpre.stat().st_size > 0 and feito:
            todas = list(_csv.reader(open(fpre, encoding="utf-8")))
            cab_p, corpo = (todas[0], todas[1:]) if todas else ([], [])
            conta = {}
            for r in corpo:
                if len(r) >= 6:
                    conta[(r[1], r[2], int(r[4]), int(r[5]))] =                         conta.get((r[1], r[2], int(r[4]), int(r[5])), 0) + 1
            esperado = {(r["montagem"], r["modelo"], int(r["target"]), int(r["fold"])):
                        int(r["n_teste"]) for r in linhas}
            quebradas = {k for k, n in esperado.items() if conta.get(k, 0) != n}
            if quebradas:
                feito -= quebradas
                viv = [r for r in corpo if len(r) >= 6 and
                       (r[1], r[2], int(r[4]), int(r[5])) not in quebradas]
                with open(fpre, "w", encoding="utf-8", newline="") as h:
                    w = _csv.writer(h); w.writerow(cab_p); w.writerows(viv)
                with open(fmet, "w", encoding="utf-8", newline="") as h:
                    w = _csv.writer(h); w.writerow(cab_existente)
                    for r in linhas:
                        k = (r["montagem"], r["modelo"], int(r["target"]), int(r["fold"]))
                        if k not in quebradas:
                            w.writerow([r.get(c, "") for c in cab_existente])
                print(f"  {len(quebradas)} celula(s) inconsistente(s) desfeita(s) para "
                      f"refazer: {sorted(quebradas)[:4]}{' ...' if len(quebradas) > 4 else ''}")
            orfas = [r for r in corpo if len(r) >= 6 and
                     (r[1], r[2], int(r[4]), int(r[5])) not in esperado]
            if orfas:
                viv = [r for r in corpo if r not in orfas and len(r) >= 6 and
                       (r[1], r[2], int(r[4]), int(r[5])) not in quebradas]
                with open(fpre, "w", encoding="utf-8", newline="") as h:
                    w = _csv.writer(h); w.writerow(cab_p); w.writerows(viv)
                print(f"  {len(orfas)} predicao(oes) orfa(s) descartada(s) "
                      f"— celula sem marcador de commit")

    CAB_PRED =["carimbo","montagem","modelo","grupo","target","fold","epoch_idx",
                "y_true","y_pred","p_rest","p_left","p_right"]
    # Correcao D-11: predicoes.csv nao tinha portao nenhum, nem de contrato nem de
    # esquema, apesar de ser a base da matriz de decisao e da analise de
    # ortogonalidade — que e a razao de B-3 existir. Mesma regra do metricas.csv.
    if fpre.exists() and fpre.stat().st_size > 0:
        _lp = _csv.reader(open(fpre, encoding="utf-8"))
        _cab_p = next(_lp, [])
        if _cab_p and _cab_p != CAB_PRED:
            raise SystemExit(f"ABORTA: {fpre} tem esquema {_cab_p} e esta run gravaria "
                             f"{CAB_PRED}. Use --out-dir novo.")
        _selos_p = {r[0] for r in _lp if r}
        if _selos_p - {SELO}:
            raise SystemExit(f"ABORTA: {fpre} tem predicoes do(s) contrato(s) "
                             f"{sorted(_selos_p)} e esta run e {SELO}. Use --out-dir novo.")

    hm = open(fmet, "a", encoding="utf-8", newline=""); wm = _csv.writer(hm)
    hp = open(fpre, "a", encoding="utf-8", newline=""); wp = _csv.writer(hp)
    if hp.tell() == 0:
        wp.writerow(CAB_PRED); hp.flush()

    # Colunas do carimbo: as condicoes DECLARADAS. As MEDIDAS entram por linha.
    SELO_COLS = ["carimbo","banda_lo","banda_hi","ordem","car","iqr_k",
                 "calib_n","fases_treino","fase2_regime","escala"]
    selo_vals = [SELO, cfg["banda_hz"][0], cfg["banda_hz"][1], cfg["ordem_filtro"],
                 int(cfg["car"]), cfg["iqr_k"], cfg["calib_n"] or 0,
                 cfg["fases_treino"], cfg["fase2_regime"], cfg["escala"]]
    MED_COLS = ["n_ch_efetivo","n_ch_modelo","n_comp_efetivo",
                "sd_entrada_uv","banda_ef_lo","banda_ef_hi"]
    verificados = set()

    t0 = time.time()
    for mt in mts:
        chs = MONTAGENS[mt]
        idx = None if chs is None else [todos.index(c) for c in chs]
        n_ch = len(todos) if chs is None else len(chs)
        nomes = list(todos) if chs is None else list(chs)
        for md in mds:
            print(f"\n{'='*62}\n  MONTAGEM {mt} ({n_ch} canais) | MODELO {md} "
                  f"| {(time.time()-t0)/60:.0f}min\n{'='*62}", flush=True)
            for gi, grupo in enumerate(grupos, 1):
                if all((mt, md, T, f) in feito for T in grupo for f in range(A.folds)):
                    continue
                pool = [s_ for s_ in range(A.subjects[0], A.subjects[1]+1)
                        if s_ not in grupo and s_ not in BAD_SUBJECTS]
                if A.pool_max and A.pool_max < len(pool):
                    pool = sorted(np.random.default_rng(A.seed).choice(
                        pool, A.pool_max, replace=False).tolist())
                print(f"  grupo {gi}/{len(grupos)} | alvos {grupo} | pool={len(pool)} "
                      f"({(time.time()-t0)/60:.0f}min)", flush=True)
                np.random.seed(A.seed); torch.manual_seed(A.seed)
                try:
                    so_mi = cfg["fases_treino"] == "mi"
                    Xm, ym, ruins_m = carrega_pool(pool, "mi", idx, cfg)
                    # No braco de fase unica o movimento executado nao e carregado:
                    # alem de nao ser usado, ele e metade do custo de carga do pool.
                    if so_mi:
                        Xr = yr = None; ruins_r = []
                    else:
                        Xr, yr, ruins_r = carrega_pool(pool, "real", idx, cfg)
                    if ruins_r or ruins_m:
                        print(f"      pool: {len(set(ruins_r) | set(ruins_m))} sujeito(s) "
                              f"fora por rejeicao excessiva {sorted(set(ruins_r) | set(ruins_m))}",
                              flush=True)
                    ts = time.time()
                    clf = CT.constroi(md, cfg)
                    if md not in verificados:
                        ruins = CT.verifica_instancia(md, clf, cfg)
                        if ruins:
                            raise SystemExit("PORTAO 2 (instancia) reprovou:\n  "
                                             + "\n  ".join(ruins))
                        verificados.add(md)
                    # Modelos que precisam saber QUAIS canais recebem, e nao apenas
                    # quantos. Hoje so o EEGSym: ele parte a entrada em dois ramos
                    # hemisfericos, e com indices fixos os ramos misturavam os lados.
                    if hasattr(clf, "set_channels"):
                        clf.set_channels(nomes)
                        if gi == 1:
                            print(f"      {clf.layout_legivel()}", flush=True)
                    if so_mi:
                        # Fase unica: `_fase1` e o ajuste COMPLETO do modelo, filtros
                        # espaciais e classificador, e aqui ele recebe imageamento.
                        # Nao da para simplesmente pular a fase 1 e chamar a fase 2:
                        # os tres classicos exigem `_phase1_fitted` e o `fit_phase2_mi`
                        # deles reajusta so o decisor, mantendo filtros que nunca
                        # teriam sido estimados.
                        _fase1(clf, Xm, ym)
                    else:
                        _fase1(clf, Xr, yr); _fase2(clf, Xm, ym)
                    print(f"      pool treinado ({time.time()-ts:.0f}s) "
                          f"— serve aos {len(grupo)} alvos do grupo", flush=True)
                    del Xr, yr, Xm, ym

                    for T in grupo:
                        if all((mt, md, T, f) in feito for f in range(A.folds)):
                            continue
                        # D-9: o alvo e a unidade de perda. Uma falha aqui custa 3
                        # celulas, nao as 15 do grupo, e o pool ja treinado continua
                        # servindo os demais alvos em vez de ser jogado fora junto.
                        try:
                            Xt, yt, keep_t = carrega(T, "mi", idx, cfg)
                        except CT.RejeicaoExcessiva as e:
                            print(f"      S{T:03d}: CELULA SUSPEITA: {e}", flush=True); continue
                        if len(np.unique(yt)) < 3 or len(Xt) < 3 * A.folds:
                            print(f"      S{T:03d}: dados insuficientes", flush=True); continue
                        skf = StratifiedKFold(A.folds, shuffle=True, random_state=A.seed)
                        for f, (itr, ite) in enumerate(skf.split(Xt, yt)):
                            if (mt, md, T, f) in feito: continue
                            ta = time.time()
                            # B-8: orcamento ABSOLUTO de calibracao. Sem ele, "menos
                            # canais" e "menos epocas de calibracao" variavam juntos,
                            # porque a retencao depende da densidade, e a queda medida
                            # nao podia ser atribuida a nenhum dos dois.
                            sel, orc_ok = CT.orcamento_calibracao(
                                yt[itr], cfg["calib_n"], A.seed + f)
                            ical = itr[sel]
                            c = copy.deepcopy(clf)
                            _calibra(c, Xt[ical], yt[ical])
                            yp = c.predict(Xt[ite])
                            # Sem try/except: saida macia e requisito de admissao
                            # (B-4), e engoli-la aqui foi como tres modelos entraram
                            # na tabela sem probabilidade nenhuma.
                            pr = c.predict_proba(Xt[ite])
                            m = compute_metrics_setup1_hierarchical(yt[ite], yp, y_proba=pr)
                            med = CT.medidas(Xt[ite], cfg["sfreq"], c)
                            if MET_COLS is None:
                                MET_COLS = sorted(m)
                                cab = (SELO_COLS +
                                       ["montagem","n_canais","modelo","grupo","target",
                                        "fold","n_pool","n_calib","n_calib_alvo",
                                        "orcamento_ok","n_teste","secs"]
                                       + MED_COLS + MET_COLS)
                                if hm.tell() == 0:
                                    wm.writerow(cab); hm.flush()
                                elif cab_existente and cab_existente != cab:
                                    # Correcao D-10: o carimbo cobre as CONDICOES, nao o
                                    # esquema do arquivo. Acrescentar uma coluna medida
                                    # sem mudar o hash passava pelo portao e as linhas
                                    # novas entravam deslocadas — o monitor leria a banda
                                    # efetiva na coluna da acuracia, sem nada falhar.
                                    faltam = set(cab) - set(cab_existente)
                                    sobram = set(cab_existente) - set(cab)
                                    raise SystemExit(
                                        f"ABORTA: {fmet} tem esquema diferente do que "
                                        f"esta run gravaria (mesmo carimbo {SELO}). "
                                        f"colunas novas: {sorted(faltam) or '-'}; "
                                        f"colunas ausentes: {sorted(sobram) or '-'}. "
                                        f"Use --out-dir novo.")
                            yv = yt[ite]
                            for j, e in enumerate(ite):
                                # B-3: `keep_t[e]` e a posicao no array ORIGINAL do
                                # sujeito. `e` sozinho e posicao no array pos-rejeicao,
                                # que retem conjuntos diferentes em cada densidade — a
                                # mesma posicao apontava para epocas diferentes em 8,
                                # 21 e 64 canais, e qualquer analise pareada por epoca
                                # comparava coisas que nao eram a mesma.
                                linha = [SELO, mt, md, gi, T, f, int(keep_t[e]),
                                         int(yv[j]), int(yp[j])]
                                linha += ([f"{pr[j,k]:.4f}" for k in range(min(3, pr.shape[1]))]
                                          if pr is not None else ["", "", ""])
                                wp.writerow(linha)
                            hp.flush()
                            # A linha de metricas e o MARCADOR DE COMMIT da celula, e
                            # por isso vai depois das predicoes: se a queda acontecer no
                            # meio da gravacao, a celula NAO consta como concluida e a
                            # retomada a refaz inteira. Na ordem inversa a celula contava
                            # como pronta com as predicoes pela metade, e o arquivo que
                            # sustenta a matriz de decisao ficava furado em silencio.
                            wm.writerow(selo_vals +
                                        [mt, n_ch, md, gi, T, f, len(pool), len(ical),
                                         cfg["calib_n"] or len(itr), int(orc_ok), len(ite),
                                         f"{time.time()-ta:.1f}"] +
                                        [f"{med[k]:.4f}" if isinstance(med[k], float) else med[k]
                                         for k in MED_COLS] +
                                        [f"{m.get(k, float('nan')):.6f}" for k in MET_COLS])
                            hm.flush()
                            print(f"      S{T:03d} fold {f}: acc3={m.get('accuracy_3cls',0):.3f}"
                                  f"{'' if orc_ok else '  [orcamento nao atingido]'}",
                                  flush=True)
                    del clf
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except CT.RejeicaoExcessiva as e:
                    # B-7: a trava de retencao e condicao de erro, nunca criterio. A
                    # celula fica de fora e sinalizada, em vez de entrar com o
                    # criterio silenciosamente trocado por "as 50% de menor amplitude".
                    print(f"      ERRO em {mt}/{md}/grupo{gi}: CELULA SUSPEITA: {e}",
                          flush=True)
                    continue
                except Exception as e:
                    print(f"      ERRO em {mt}/{md}/grupo{gi}: {type(e).__name__}: {e}",
                          flush=True)
                    continue
    hm.close(); hp.close()
    print(f"\nBENCH_MONTAGENS_DONE ({(time.time()-t0)/60:.0f}min)")


if __name__ == "__main__":
    main()
