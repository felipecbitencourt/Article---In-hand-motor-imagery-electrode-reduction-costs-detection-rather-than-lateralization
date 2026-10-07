# -*- coding: utf-8 -*-
"""Figuras do artigo, sobre results/bench_contrato_mi (carimbo 5cb6c6358c).

PUBLIC PACKAGE NOTE (2026-10). This is `artigo-montagens/figuras.py` from the
research repository, kept as close to the original as possible. Changes:
  - paths point to this package (results/ and figures/), RESULTS_DIR overrides;
  - every figure is saved as PNG (400 dpi), SVG and PDF;
  - Figure 2: STIA-Net is no longer drawn in grey (it was confused with the
    light grey lines of the non-highlighted decoders), and an in-figure legend
    names the light grey lines;
  - Figure S2 (new): the same two panels with all eight decoders labelled;
  - English labels by default (IDIOMA=pt for the Portuguese version).
Usage, from the package root:  python analysis/figures.py
Figure 1 reads one recording (S001R04) for the electrode positions, so it needs
the PhysioNet data in data_base_physionet/ (see README).

FONTE CANONICA: o braco de treino em fase unica, so imageamento. O outro
braco, `bench_contrato`, treina tambem com movimento executado e dobra o
pool de 8 542 para ~17 085 epocas — e nao rende: a diferenca em deteccao e
de +0,004 sobre as oito arquiteturas e nada sobrevive a Holm. A apresentacao
ja revisada usa este braco, e o artigo passou a usa-lo tambem para os dois
documentos nao divergirem.

Duas figuras, na ordem em que aparecem no artigo. A 1 mostra o que as montagens
SAO, com os 64 eletrodos numa cabeca e uma cor por degrau da escada; a 2 mostra o
que a reducao de densidade custa.

Formato herdado do artigo anterior: largas e baixas, fonte 7, dpi 400, de modo
que o builder as encaixe em 15 cm de largura.

ESCOPO. O artigo trata so de reducao de densidade. As montagens de controle
(posterior8, sem_fz, ocular4) e o degrau de 21 saem da analise: aquelas
respondem "quais" em vez de "quantos", e o 21 foi retirado para consolidar seis
degraus. Os dados delas seguem em results/bench_contrato_mi e podem ser retomados.
"""
import io, os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle, Polygon

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(BASE, ".."))
FIG = os.path.join(PKG, "figures"); os.makedirs(FIG, exist_ok=True)
EXP = os.path.join(PKG, "experiments", "hierarquico-8")
RES = os.environ.get("RESULTS_DIR", os.path.join(PKG, "results", "bench_contrato_mi"))


def salva(nome, **kw):
    """PNG at 400 dpi for the paper builder, plus SVG and PDF (vector)."""
    for ext in ("png", "svg", "pdf"):
        plt.savefig(f"{FIG}/{nome}.{ext}", dpi=400, **kw)
    plt.close()

d = pd.concat([pd.read_csv(f, dtype={"montagem": str, "carimbo": str})
               for f in glob.glob(os.path.join(RES, "*", "metricas.csv"))], ignore_index=True)
assert d.carimbo.nunique() == 1, "mistura de contratos"
d["lat"] = d.accuracy_lateralization_given_mi - 0.5 * d.sensitivity_detection

ESCADA = ["2", "4", "8", "16", "32", "64"]
NCH = {c: int(c) for c in ESCADA}
NOME = {"csp_lda": "CSP+LDA", "fbcsp_svm": "FBCSP+SVM", "riemannian": "Riemannian",
        "eegnet": "EEGNet", "eegsym": "EEGSym", "shallow_convnet": "ShallowConvNet",
        "deepcnn": "Adaptive Deep CNN", "stia_net": "STIA-Net"}
# Destaques escolhidos pelo que a curva precisa mostrar: o que mais cai, o melhor
# detector e o que nao cai porque opera perto do piso em toda densidade.
# Cor FIXA por modelo, usada onde quer que ele seja destacado. Um mesmo modelo
# com cores diferentes entre figuras faria o leitor perder o fio.
# STIA-Net was "#7a7a7a" (grey) in the submitted version. A reviewer could not tell
# it from the light grey lines of the five non-highlighted decoders, so it now gets
# vermillion (Okabe-Ito), distinct from the blue and purple highlights and safe for
# colour-blind readers.
COR = {"eegsym": "#a83232", "riemannian": "#1f4e79", "csp_lda": "#e08214",
       "deepcnn": "#5b2c8d", "stia_net": "#D55E00", "eegnet": "#2e7d32"}
CINZA_OUTROS = "#c8c8c8"   # non-highlighted decoders in Figure 2
# Quem cada figura destaca, escolhido pelo que a curva daquela figura precisa
# mostrar, e nao um conjunto fixo: na Figura 2 o eixo e quem cai mais; na 3, o
# melhor detector, o melhor lateralizador e o que nao decodifica.
DEST_DL = ["riemannian", "deepcnn", "stia_net"]
# Rampa sequencial para a escada, e nao seis matizes distintos: a escada e
# ORDENADA, e cores categoricas fariam 64 parecer tao distante de 32 quanto de 2.
#
# VIRIDIS no lugar de Blues, e o motivo e distincao e nao estetica. O Blues varia
# so em luminosidade, entao os seis degraus se separavam por um unico eixo
# perceptual e os do meio ficavam quase iguais — e ele ja usava a faixa de 0,07 a
# 0,99, ou seja, esticar nao era opcao. O viridis varia matiz E luminosidade, e e
# perceptualmente uniforme: a mesma distancia numerica rende a mesma distancia
# visual em qualquer trecho da escala. Continua ORDENADO, entao o argumento
# acima segue valendo, e sobrevive a impressao em tons de cinza e a daltonismo.
#
# A direcao e a de antes: 2 eletrodos escuro, 64 claro. A montagem pequena e a
# que o artigo discute, e ela tem de saltar sobre as 32 bolinhas do arranjo
# completo, que devem recuar para o fundo.
CORE = {c: cm.viridis(v) for c, v in zip(ESCADA, [0.02, 0.24, 0.44, 0.62, 0.80, 0.96])}

# Rotulos por idioma. O artigo tem versao em portugues e em ingles, e figura com
# rotulo no idioma errado e erro de submissao, nao detalhe.
IDIOMA = os.environ.get("IDIOMA", "en")
L = {
 "pt": dict(ele="eletrodos", kap="kappa de 3 classes", fra="fração do valor em 64",
            det="detecção acima do acaso", lat="lateralização acima do piso",
            cls=["repouso", "esquerda", "direita"],
            verd="verdade", pred="predito",
            m_mont="montagem", m_add="acrescenta", m_ele="eletrodos",
            m_resto="os {n} restantes do arranjo 10-10", suf="",
            det_ax="detecção", lat_ax="lateralização",
            t3a="(a) a detecção sobe até 64",
            t3b="(b) a lateralização satura em 8",
            outros="os outros cinco decodificadores", piso_leg="acaso (a) e piso (b)"),
 "en": dict(ele="electrodes", kap="three class kappa", fra="fraction of the value at 64",
            det="detection above chance", lat="lateralization above floor",
            cls=["rest", "left", "right"],
            verd="true", pred="predicted",
            m_mont="montage", m_add="adds", m_ele="electrodes",
            m_resto="the remaining {n} of the 10-10 array", suf="_en",
            det_ax="detection", lat_ax="lateralization",
            t3a="(a) detection rises to 64",
            t3b="(b) lateralization saturates at 8",
            outros="the other five decoders", piso_leg="chance (a) and floor (b)"),
}[IDIOMA]

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.3, "axes.linewidth": 0.6})


def curva(met):
    p = d.pivot_table(index="modelo", columns="montagem", values=met)
    return {m: [p.loc[m, c] for c in ESCADA] for m in p.index}


def rotula_pontas(a_, itens, x_fim, sep=7.2):
    """Nomeia o fim de cada curva destacada sem deixar rotulos um sobre o outro.

    Duas curvas que terminam quase no mesmo valor imprimiam os nomes sobrepostos
    e o leitor perdia os dois: EEGNet fecha em 0,376 e Riemannian em 0,370, e a
    2 pt de distancia os dois viravam borrao. O empurrao e feito em PONTOS e nao
    em unidades de dado, porque a colisao e do TEXTO e nao da curva, e por isso
    depende da altura do eixo em pontos e nao da escala do eixo. A linha continua
    marcando o valor verdadeiro; so o rotulo se desloca.

    A ordem importa: chamar ANTES do tight_layout, com um `draw` previo para o
    eixo ter geometria. O rotulo fica FORA da curva e entra no tightbbox do eixo,
    entao o tight_layout encolhe o eixo o bastante para ele caber; ao contrario,
    "Adaptive Deep CNN" era cortado pela borda direita da figura.
    """
    a_.figure.canvas.draw()
    alt_pt = a_.get_window_extent().height / a_.figure.dpi * 72
    lo, hi = a_.get_ylim()
    ppd = alt_pt / (hi - lo)                       # pontos por unidade de dado
    itens = sorted(itens, key=lambda t_: t_[1])    # de baixo para cima
    ys = [v * ppd for _, v, _ in itens]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + sep)
    for (nome, v, cor), y in zip(itens, ys):
        a_.annotate(nome, (x_fim, v), xytext=(3, y - v * ppd - 1),
                    textcoords="offset points", color=cor, fontsize=6.2,
                    va="center")


# ---------------------------------------------------------------- Figura 3 --
# A MATRIZ DE DECISAO, e nao mais as curvas de densidade agregadas. A figura que
# estava aqui mostrava os dois niveis em valor absoluto e em fracao retida; a
# Figura 2 ao lado ja mostra os MESMOS dois niveis, decodificador a
# decodificador, e as duas acabaram sendo vistas do mesmo par de curvas.
#
# A matriz responde o que curva nenhuma responde: PARA ONDE a epoca vai quando
# erra. Uma epoca de esquerda pode virar direita, que e falha de lateralizacao,
# ou virar repouso, que e falha de deteccao. Sao problemas distintos, pedem
# correcoes distintas, e a acuracia soma os dois num numero so.
#
# E e aqui que a tese do artigo aparece nos proprios erros: de 64 para 8 a troca
# de lado fica praticamente parada (0,282 para 0,264) enquanto imageamento dito
# repouso sobe (0,271 para 0,309). A reducao ate 8 eletrodos compra erro de
# DETECCAO, nao de lateralizacao.
DENS_M = ["64", "8", "2"]                      # na ordem em que se reduz
CLASSES = ["rest", "left", "right"]
fig, ax = plt.subplots(1, 3, figsize=(5.9, 1.52))
for a_, c in zip(ax, DENS_M):
    s = d[d.montagem == c]
    M = np.array([[s[f"cm_{va}_pred_{vp}"].sum() for vp in CLASSES]
                  for va in CLASSES], dtype=float)
    N = M / M.sum(axis=1, keepdims=True)       # linha = verdade, soma 1
    a_.imshow(N, cmap="BuGn", vmin=0.15, vmax=0.70, aspect="auto")
    for r in range(3):
        for cc in range(3):
            # texto claro so onde o fundo e escuro, senao some
            a_.text(cc, r, f"{N[r, cc]:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=6.0, color="white" if N[r, cc] > 0.48 else "#1a1a1a")
    a_.set_xticks(range(3)); a_.set_xticklabels(L["cls"], fontsize=6.2)
    a_.set_yticks(range(3)); a_.set_yticklabels(L["cls"], fontsize=6.2)
    a_.set_title(f"{c} {L['m_ele']}", loc="left", fontsize=6.9)
    a_.tick_params(length=0)
    for sp in a_.spines.values():
        sp.set_visible(False)
ax[0].set_ylabel(L["verd"]); ax[1].set_xlabel(L["pred"])
plt.tight_layout(pad=0.3)
salva(f"figS3_confusion{L['suf']}")


# ---------------------------------------------------------------- Figura 2 --
# DETECCAO E LATERALIZACAO SEPARADAS, decodificador a decodificador. A Figura 2b
# ja mostra que a perda se aloja na deteccao, mas em AGREGADO: nao diz se o
# padrao vale para todos ou se e media de comportamentos opostos. Vale para
# todos, e a forma das duas familias e qualitativamente distinta — que e a
# afirmacao do artigo, e ela merece ser vista e nao so lida.
x = [NCH[c] for c in ESCADA]          # o eixo da escada, em numero de canais
fig, ax = plt.subplots(1, 2, figsize=(5.9, 1.50))
pontas3 = {}
for a_, met, dest, rot, piso in (
        (ax[0], "bca_detection", DEST_DL, L["det_ax"], 0.5),
        (ax[1], "lat", DEST_DL, L["lat_ax"], 0.0)):
    cur = curva(met)
    pontas3[a_] = []
    for m, v in sorted(cur.items(), key=lambda t_: -t_[1][-1]):
        c = COR[m] if m in dest else CINZA_OUTROS
        a_.plot(x, v, "-o", color=c, lw=1.5 if m in dest else 0.8,
                ms=2.6 if m in dest else 1.6, zorder=3 if m in dest else 1)
        if m in dest:
            pontas3[a_].append((NOME[m], v[-1], c))
    a_.axhline(piso, color="#888", ls=":", lw=0.7)
    a_.set_xscale("log", base=2); a_.set_xticks(x); a_.set_xticklabels(ESCADA)
    a_.set_xlabel(L["ele"]); a_.set_ylabel(rot)
    a_.set_xlim(1.8, 200); a_.spines[["top", "right"]].set_visible(False)
ax[0].set_title(L["t3a"], loc="left", fontsize=6.9)
ax[1].set_title(L["t3b"], loc="left", fontsize=6.9)
for a_, itens in pontas3.items():
    rotula_pontas(a_, itens, x[-1])
# Reviewer 2 (ERAMIA-RS 2026): the light grey lines were not identified. They are
# the five decoders that are not highlighted, and are now named inside the figure.
# The dotted line (chance in (a), floor of lateralization in (b)) is left unlabelled.
from matplotlib.lines import Line2D
ax[0].legend(handles=[
    Line2D([], [], color=CINZA_OUTROS, lw=0.8, marker="o", ms=1.6, label=L["outros"])],
    loc="upper left", frameon=False, fontsize=5.6, handlelength=1.8,
    borderaxespad=0.2, labelspacing=0.3)
# w_pad maior que o das outras figuras: "Adaptive Deep CNN" e o rotulo mais
# comprido do artigo e, com o padding padrao, encostava no rotulo do eixo y do
# painel (b) com menos de meio milimetro de folga no tamanho impresso.
plt.tight_layout(pad=0.3, w_pad=2.2)
salva(f"fig2_niveis{L['suf']}")

# --------------------------------------------------------------- Figure S2 --
# Supplementary, for the public repository: the same two panels with every decoder
# coloured and named, so no line has to be identified from the caption.
COR_S = dict(COR, fbcsp_svm="#56B4E9", shallow_convnet="#CC79A7")
MARCA = {"riemannian": "o", "deepcnn": "s", "stia_net": "X", "csp_lda": "^",
         "fbcsp_svm": "v", "eegnet": "D", "eegsym": "P", "shallow_convnet": "*"}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 1.95))
for a_, met, rot, piso in ((ax[0], "bca_detection", L["det_ax"], 0.5),
                           (ax[1], "lat", L["lat_ax"], 0.0)):
    for m, v in sorted(curva(met).items(), key=lambda t_: -t_[1][-1]):
        a_.plot(x, v, "-", marker=MARCA[m], color=COR_S[m], lw=1.0, ms=2.8, label=NOME[m])
    a_.axhline(piso, color="#888", ls=":", lw=0.7)
    a_.set_xscale("log", base=2); a_.set_xticks(x); a_.set_xticklabels(ESCADA)
    a_.set_xlabel(L["ele"]); a_.set_ylabel(rot)
    a_.set_xlim(1.8, 80); a_.spines[["top", "right"]].set_visible(False)
ax[0].set_title(L["t3a"], loc="left", fontsize=6.9)
ax[1].set_title(L["t3b"], loc="left", fontsize=6.9)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", frameon=False, fontsize=6.0, handlelength=2.0)
plt.tight_layout(pad=0.3, w_pad=2.0, rect=(0, 0, 0.80, 1))
salva(f"figS2_all_decoders{L['suf']}")

# ---------------------------------------------------------------- Figura 1 --
# As montagens, na cabeca. Uma tabela com as listas de canais diria o mesmo em
# principio, mas ninguem enxerga "faixa central bilateral" numa lista de siglas;
# e o aninhamento, que e a propriedade de que depende atribuir a diferenca a
# densidade, so fica evidente quando cada degrau aparece contido no seguinte.
import warnings; warnings.filterwarnings("ignore")
sys.path[:0] = [os.path.abspath(os.path.join(BASE, "..")),
                os.path.abspath(os.path.join(BASE, "..", "modelos")),
                os.path.abspath(EXP)]
import mne
from mne.channels.layout import _find_topomap_coords
from core.data_loader import load_run
import _bench_montagens as B

raw = load_run(1, 4); mne.datasets.eegbci.standardize(raw)
nomes = raw.info["ch_names"]
raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
xy = _find_topomap_coords(raw.info, picks=range(len(nomes)))

# SIMETRIZAR E CENTRAR. A projecao do `_find_topomap_coords` nao sai simetrica:
# 19 dos 64 canais ficavam sem espelho em -x dentro de 0,02, e a nuvem inteira
# saia deslocada (x de -0,772 a 0,738). Numa cabeca de EEG isso le como erro de
# posicionamento, porque o arranjo 10-10 E simetrico por construcao.
#
# O conserto e medio com o espelho: para cada canal, a posicao final e a media
# entre a dele e a reflexao do seu par. Canais da linha media viram o proprio
# par, entao a media os empurra para x = 0 exatamente, que e onde devem estar.
xy = xy - xy.mean(axis=0)
espelho = np.array([[-1.0, 1.0]])
par = [int(np.argmin(np.linalg.norm(xy - xy[i] * espelho, axis=1)))
       for i in range(len(xy))]
xy = (xy + xy[par] * espelho) / 2.0
xy = xy / np.abs(xy).max() * 0.85

entra = {}
for c in ESCADA:
    for ch in (B.MONTAGENS[c] or nomes):
        entra.setdefault(ch, c)
assert len(entra) == 64, f"escada nao cobre os 64 canais: {len(entra)}"

# Proporcao ditada pelo builder, nao pelo gosto: ele limita a figura a 4,2 cm de
# altura e so entao a alarga ate os 15 cm da mancha, entao qualquer figura com
# razao abaixo de 15/4,2 = 3,6 entra estreita e leva os rotulos junto. Numa versao
# anterior, com razao 2,55, a cabeca chegava a 10,7 cm e a fonte efetiva dos nomes
# dos eletrodos caia para cerca de 4 pt.
# Alargada de 5,9 para 6,15 ao tirar os rotulos: sem eles o painel da cabeca
# encolhe na horizontal, o `bbox_inches="tight"` corta mais largura que altura, e
# a razao final caiu de 3,9 para 3,51 — abaixo dos 15/4,2 = 3,571 que o builder
# exige. Passar disso nao quebra, so faz a figura entrar com menos de 15 cm.
fig = plt.figure(figsize=(6.15, 1.52))
ax = fig.add_axes([0.005, 0.01, 0.28, 0.98]); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0, 0), 1.0, fc="none", ec="#555", lw=1.0, zorder=1))
ax.add_patch(Polygon([[-0.12, 0.99], [0, 1.15], [0.12, 0.99]], fc="none", ec="#555",
                     lw=1.0, zorder=1))
for s in (-1, 1):
    ax.add_patch(Polygon([[s*0.99, 0.12], [s*1.10, 0.06], [s*1.10, -0.06], [s*0.99, -0.12]],
                         fc="none", ec="#555", lw=1.0, zorder=1))
for i, ch in enumerate(nomes):
    # TAMANHO UNICO. A cor sozinha carrega o degrau, e ela foi feita para isso:
    # o viridis separa o par de vizinhos mais proximo por 34,1 em Lab, contra
    # 13,7 do Blues anterior. Tamanho variavel acrescentaria um segundo canal
    # redundante e, pior, faria o leitor procurar significado numa diferenca que
    # nao existe alem do nivel.
    #
    # O VALOR E MEDIDO. A cabeca e limitada pela ALTURA do painel por causa do
    # aspect igual, entao ha ~45 pt por unidade de dado; o par mais apertado do
    # arranjo 10-10 e P7-P5, a 3,89 pt. Com s=13 o marcador tem 3,61 pt de
    # diametro e sobra 0,28 pt. Alargar o painel nao ajuda: o aspecto o trava na
    # altura, entao 13 e o teto e nao uma preferencia.
    TAM = 13
    c = entra[ch]
    ax.scatter(*xy[i], s=TAM, c=[CORE[c]], ec="#222", lw=0.3,
               zorder={"2": 6, "4": 5, "8": 4, "16": 3, "32": 2, "64": 1}[c])
# SEM ROTULO DE ELETRODO. A tabela ao lado ja diz quais sao, nome por nome, e
# repetir aqui custava caro: os rotulos exigiam raio 1,14 mais folga de texto, o
# que empurrava o limite do eixo para 1,32 e encolhia a cabeca a menos de dois
# tercos da caixa. Sem eles a cabeca ocupa o espaco todo e os pontos ficam
# maiores, que e o que a figura precisa mostrar — onde os eletrodos estao, nao
# como se chamam.
ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)

axr = fig.add_axes([0.30, 0.01, 0.69, 0.98]); axr.axis("off")
axr.set_xlim(0, 1); axr.set_ylim(0, 1)
# As linhas sao distribuidas por contagem, e nao por deslocamentos somados a mao:
# a montagem de 32 quebra em duas linhas e a versao anterior descontava isso do
# espacamento seguinte, de modo que mudar a largura de quebra desalinhava a lista.
#
# A QUEBRA E MEDIDA, e nao contada em caracteres. Com `textwrap.wrap` num limite
# de 62 caracteres, a linha de 32 quebrava com 52 pt de coluna ainda vazios: a
# lista e quase toda de caracteres estreitos (virgulas, digitos, "1"), entao a
# contagem de caracteres superestima a largura de quem tem muitos deles. Aqui a
# largura de cada candidata e medida na fonte real, contra a coluna real.
COL_X = 0.27                                   # onde a coluna "acrescenta" comeca
fig.canvas.draw()                              # o renderizador precisa existir
_rend = fig.canvas.get_renderer()
_larg_ax_pt = axr.get_window_extent().width / fig.dpi * 72
LIM_PT = (1.0 - COL_X) * _larg_ax_pt


def _mede(s, fs=5.7):
    t_ = fig.text(0, 0, s, fontsize=fs)
    w = t_.get_window_extent(_rend).width / fig.dpi * 72
    t_.remove()
    return w


def quebra(txt, fs=5.7):
    """Quebra guloso por largura medida. Os tokens sao as palavras, o que serve
    tanto para a lista separada por virgula quanto para a frase do degrau de 64."""
    linhas, atual = [], []
    for tok in txt.split(" "):
        if atual and _mede(" ".join(atual + [tok]), fs) > LIM_PT:
            linhas.append(" ".join(atual)); atual = [tok]
        else:
            atual.append(tok)
    linhas.append(" ".join(atual))
    return linhas


blocos = []
vistos = set()
for c in ESCADA:
    novos = [x_ for x_ in (B.MONTAGENS[c] or nomes) if x_ not in vistos]; vistos |= set(novos)
    txt = ", ".join(novos) if c != "64" else L["m_resto"].format(n=len(novos))
    blocos.append((c, quebra(txt)))
n_linhas = 1 + sum(len(b[1]) for b in blocos)      # cabecalho mais o corpo
passo = 0.97 / (n_linhas + len(blocos) * 0.45)     # respiro entre blocos
y = 1.0 - passo
axr.text(0.0, y, L["m_mont"], fontsize=6.4, weight="bold", va="baseline")
axr.text(0.27, y, L["m_add"], fontsize=6.4, weight="bold", va="baseline")
y -= passo * 1.45
sep = []                                   # onde cada degrau termina
for c, linhas in blocos:
    axr.scatter([0.018], [y + passo*0.22], s=26, c=[CORE[c]], ec="#222", lw=0.45,
                transform=axr.transAxes, clip_on=False)
    axr.text(0.06, y, f"{c} {L['m_ele']}", fontsize=6.2, va="baseline")
    for j, ln in enumerate(linhas):
        axr.text(0.27, y - j*passo, ln, fontsize=5.7, va="baseline")
    sep.append(y - (len(linhas) - 1) * passo)
    y -= passo * (len(linhas) + 0.45)

# REGUAS. Sem elas a lista era so texto solto: com a montagem de 32 quebrando em
# duas linhas, nao havia como ver de relance onde um degrau acaba e o proximo
# comeca, e o olho lia "T7, T8" como se fosse do degrau de 64. O desenho segue o
# das tabelas do artigo, do template SBC: regua acima, regua sob o cabecalho e
# regua no fim, em preto; entre os degraus, fio cinza claro, que separa sem
# competir com o texto.
#
# As alturas sao dadas em fracao de `passo`, e nao em valores fixos, porque
# `passo` ja e a entrelinha da lista: 0,55 abaixo da linha de base cai no meio do
# branco entre uma linha e a proxima em qualquer numero de linhas.
def regua(yy, lw, cor):
    axr.axhline(yy, xmin=0.0, xmax=1.0, color=cor, lw=lw, zorder=0,
                clip_on=False)


y_cab = 1.0 - passo
regua(y_cab + passo * 0.72, 0.8, "#222")                 # topo
regua(y_cab - passo * 0.55, 0.5, "#222")                 # sob o cabecalho
for s_ in sep[:-1]:
    regua(s_ - passo * 0.55, 0.35, "#c9c9c9")            # entre degraus
regua(sep[-1] - passo * 0.55, 0.8, "#222")               # fim
salva(f"fig1_montagens{L['suf']}", bbox_inches="tight", facecolor="white")

for f in (f"fig1_montagens{L['suf']}", f"fig2_niveis{L['suf']}",
          f"figS3_confusion{L['suf']}", f"figS2_all_decoders{L['suf']}"):
    from PIL import Image
    im = Image.open(f"{FIG}/{f}.png")
    print(f"  {f}.png  {im.size[0]}x{im.size[1]} px, razao {im.size[0]/im.size[1]:.2f}"
          f"  -> {15.0:.0f} cm x {15.0*im.size[1]/im.size[0]:.1f} cm no artigo")
