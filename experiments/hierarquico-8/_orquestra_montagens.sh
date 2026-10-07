#!/usr/bin/env bash
# Orquestrador do benchmark por montagem, em DUAS CORRENTES INDEPENDENTES.
#
# Cada corrente percorre TODAS as fases por conta propria, sem barreira entre elas.
# A versao anterior sincronizava as duas ao fim de cada fase, e como os classicos
# sao apenas 15% do trabalho, a corrente de CPU terminava cedo e ficava ociosa
# esperando a de GPU — desperdicando justamente o recurso que a paralelizacao
# pretendia aproveitar.
#
# Por que duas e nao mais: os perfis sao complementares. Os classicos sao
# scikit-learn/pyriemann, puro CPU e monothread; os profundos usam GPU. Medido com
# um so processo: GPU a 26% (45 W de 180) e CPU a 10-18% de 16 nucleos.
#
# Por que NAO paralelizar os profundos entre si: sao 85% do trabalho, e foi ai que
# o OOM de agosto ocorreu (stia_net comecando com 12,4 GB de resto do deepcnn).
#
# A ordem das montagens dentro de cada corrente e a mesma e reflete prioridade:
# controles primeiro (baratos e podem reorientar o artigo), depois as duas pontas
# (respondem a pergunta central), depois a curva. Se a run cair, o que ficou pronto
# e o que mais importa.
#
# Cada corrente grava em seu proprio diretorio: dois processos escrevendo no mesmo
# CSV intercalariam linhas. O monitor le os dois e concatena.
#
# Retomada: o runner reconstroi o estado de (montagem, modelo, alvo, fold) do CSV,
# descarregado a cada fold. Reexecutar continua de onde parou.
set -u
# Interpretador: por padrao o venv do proprio repositorio, resolvido a partir da
# localizacao deste script. Assim o orquestrador roda em qualquer maquina sem
# edicao, que era uma pegadinha da versao anterior: o caminho estava fixo na
# maquina de desenvolvimento e falhava em silencio noutra.
# Sobreponivel:  PY=/caminho/python.exe bash _orquestra_montagens.sh
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-$RAIZ/.venv312/Scripts/python.exe}"
[ -x "$PY" ] || PY="${PY%.exe}"          # em Linux/macOS o binario nao tem .exe
if [ ! -x "$PY" ]; then
  echo "ABORTA: interpretador nao encontrado."
  echo "  procurei em: $RAIZ/.venv312/Scripts/python.exe"
  echo "  aponte o seu com:  PY=/caminho/para/python.exe bash $0"
  exit 1
fi
cd "$(dirname "$0")" || exit 1
echo "### interpretador: $PY"

ML="csp_lda fbcsp_svm riemannian"
DEEP="eegnet eegsym shallow_convnet deepcnn stia_net"
ORDEM="posterior8 sem_fz ocular4 8 64 2 4 3 16 21 32"

# Diretorio NOVO. As dez correcoes bloqueantes mudam o carimbo do contrato, e o
# runner se recusa a anexar a um CSV de outro contrato (correcao B-10) — resultado
# antigo e resultado novo nao pertencem a mesma tabela e nao devem parecer que sim.
# results/bench_montagens/ fica intacto, como registro do que foi medido antes.
# Braco de treino, escolhido pela variavel FASES. "real+mi" (padrao) e o desenho
# original, com movimento executado antes do imageamento; "mi" treina uma vez so,
# sobre imageamento. Cada braco grava em seu proprio diretorio e carrega carimbo
# proprio, entao os dois nunca entram na mesma tabela.
#   bash _orquestra_montagens.sh              -> braco original
#   FASES=mi bash _orquestra_montagens.sh     -> braco de fase unica
FASES="${FASES:-real+mi}"
if [ "$FASES" = "mi" ]; then
  BASE="results/bench_contrato_mi"
  EXTRA_FASES="--fases mi"
else
  BASE="results/bench_contrato"
  EXTRA_FASES=""
fi
echo "### braco: $FASES  ->  $BASE"

# Modelos cuja leitura fiel ao artigo de base pede banda estreita, e que por isso
# rodam tambem como CELULA GEMEA em 8-30 Hz (secao 4.2 do contrato). Divergir e
# permitido, nao e gratis: o braco primario roda na banda do contrato e a leitura do
# artigo roda ao lado, com carimbo proprio, em diretorio proprio.
GEMEA_ML="riemannian"
GEMEA_DEEP="stia_net"

# Uma corrente: para cada montagem, roda seus modelos em sequencia, um processo por
# modelo (isolamento de memoria, que foi o que resolveu o OOM de 64 canais).
corrente () {
  local rotulo="$1"; local modelos="$2"; local extra="${3:-}"; local dir="$BASE/$rotulo"
  local falhas=0 ok=0 rc=0
  mkdir -p "$dir"
  for mt in $ORDEM; do
    for md in $modelos; do
      echo "  [$rotulo] $mt / $md | $(date '+%H:%M:%S')"
      "$PY" -u _bench_montagens.py --montagens "$mt" --modelos "$md" \
            --out-dir "$dir" $EXTRA_FASES $extra >> "$BASE/$rotulo.log" 2>&1
      rc=$?
      # Correcao D-14: o codigo de saida era ignorado. Uma grade que abortasse
      # inteira — por exemplo com o diretorio ja ocupado por outro carimbo, que o
      # runner recusa desde B-10 — imprimia CORRENTE CONCLUIDA em segundos e a
      # noite se perdia sem nenhum sinal.
      if [ $rc -ne 0 ]; then
        echo "  [$rotulo] FALHA rc=$rc em $mt/$md — ver $BASE/$rotulo.log"
        falhas=$((falhas + 1))
        # Aborta cedo se NADA rodou ainda: tres falhas sem nenhum sucesso quase
        # sempre tem causa comum (contrato, caminho, ambiente), e insistir gasta a
        # noite reproduzindo o mesmo erro 88 vezes.
        if [ $falhas -ge 3 ] && [ $ok -eq 0 ]; then
          echo "  [$rotulo] ABORTA: $falhas falhas seguidas e nenhum sucesso"
          return 1
        fi
      else
        ok=$((ok + 1))
      fi
    done
  done
  echo "  [$rotulo] CORRENTE CONCLUIDA $(date '+%d/%m %H:%M:%S') — $ok ok, $falhas falha(s)"
  [ $falhas -eq 0 ]
}

echo "############ inicio $(date '+%d/%m %H:%M:%S') ############"
corrente ml   "$ML"   &
pid_ml=$!
corrente deep "$DEEP" &
pid_deep=$!

wait $pid_ml;    rc_ml=$?
# As gemeas de CPU entram assim que a corrente de CPU vaga, escondidas atras da
# fila de GPU: os tres metodos de covariancia custam 0,10 h por celula.
corrente _gemeas/ml "$GEMEA_ML" "--banda 8 30" &
pid_gml=$!

wait $pid_deep;  rc_deep=$?
corrente _gemeas/deep "$GEMEA_DEEP" "--banda 8 30" &
pid_gdeep=$!

wait $pid_gml;   rc_gml=$?
wait $pid_gdeep; rc_gdeep=$?

echo ""
total_rc=$((rc_ml + rc_deep + rc_gml + rc_gdeep))
if [ $total_rc -ne 0 ]; then
  echo "ORQUESTRA_MONTAGENS_COM_FALHAS $(date '+%d/%m %H:%M:%S')"
  echo "  ml=$rc_ml deep=$rc_deep gemea_ml=$rc_gml gemea_deep=$rc_gdeep"
  echo "  leia os .log em $BASE antes de usar qualquer CSV desta run"
else
  echo "ORQUESTRA_MONTAGENS_DONE $(date '+%d/%m %H:%M:%S')"
fi
# As gemeas ficam sob _gemeas/, e `csvs_ativos` ja pula diretorios
# iniciados por `_`: o monitor ve o contrato primario sem misturar carimbos.
"$PY" _monitor_montagens.py --dir "$BASE" 2>/dev/null | tail -5
echo "-- gemeas (banda 8-30):"
"$PY" _monitor_montagens.py --dir "$BASE/_gemeas" --modelos riemannian stia_net 2>/dev/null | tail -4
