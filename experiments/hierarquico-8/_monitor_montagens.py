# -*- coding: utf-8 -*-
"""Monitor do benchmark por montagem: progresso e ETA, atualizados a cada sujeito.

Le results/bench_montagens/metricas.csv, descarregado ao fim de cada fold, entao
reflete o estado real mesmo se a run tiver caido. Nao toca na GPU.

O ETA nao usa media global: o custo varia muito entre combinacoes (o
ShallowConvNet em 64 canais e ~30x o CSP+LDA em 2). Para cada combinacao ja
iniciada usa-se o ritmo medido nela; para as ainda nao iniciadas, extrapola-se do
custo por canal observado nas concluidas. A estimativa melhora a cada sujeito.

Detecta parada: se o CSV nao cresce entre duas leituras do modo --watch, avisa.
Isso importa porque o processo sobrevive ao fim da sessao do agente, e nenhuma
notificacao chega quando ele termina ou morre.

Uso:
  python _monitor_montagens.py --montagens todas
  python _monitor_montagens.py --watch                  # atualiza a cada 5 min
  python _monitor_montagens.py --watch --intervalo 60   # a cada minuto
"""
from __future__ import annotations
import sys, io, os, re, time, argparse
from datetime import datetime, timedelta
from pathlib import Path

# line_buffering=True: sem isso a camada de texto adiciona um buffer proprio que
# anula o -u, e a saida se perde se o processo for encerrado (Ctrl+C, timeout).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
EXP = Path(__file__).resolve().parent

ESCADA = ["64", "32", "21", "16", "8", "4", "2"]
ACIMA_DO_ACASO = ["eegnet", "shallow_convnet", "riemannian", "fbcsp_svm", "csp_lda"]
TODOS_8 = ["csp_lda", "fbcsp_svm", "riemannian", "eegnet", "eegsym",
           "shallow_convnet", "deepcnn", "stia_net"]
N_CANAIS = {"64": 64, "32": 32, "21": 21, "16": 16, "8": 8, "4": 4, "3": 3, "2": 2,
            "sem_fz": 8, "posterior8": 8, "ocular4": 4}


def csvs_ativos(raiz: Path, nome: str) -> list[Path]:
    """Varre os CSVs ignorando diretorios iniciados por `_`.

    Esses guardam resultados arquivados de definicoes ANTIGAS de montagem — hoje
    `_16_assimetrico/`, do 16 antes de virar simetrico. Trazem o mesmo rotulo de
    montagem com canais diferentes, entao concatena-los misturaria dois conjuntos de
    dados de entrada sob um so nome.
    """
    return [f for f in raiz.rglob(nome)
            if not any(p.startswith("_") for p in f.relative_to(raiz).parts[:-1])]


def relatorio(A, anterior: int | None) -> int:
    """Imprime o estado e devolve o numero de celulas, para detectar parada."""
    import pandas as pd
    mts = ESCADA if A.montagens == ["todas"] else A.montagens
    mds = (ACIMA_DO_ACASO if A.modelos == ["acima_do_acaso"]
           else TODOS_8 if A.modelos == ["todos"] else A.modelos)

    # As duas correntes (ml e deep) gravam em diretorios separados, para nao
    # intercalar linhas no mesmo arquivo. O monitor le todos e concatena.
    fms = sorted(csvs_ativos(Path(A.dir), "metricas.csv"))
    agora = datetime.now().strftime("%d/%m %H:%M:%S")
    if not fms:
        print(f"[{agora}] ainda sem metricas em {A.dir}")
        return 0
    # Um CSV recem-criado fica vazio ate o primeiro fold fechar (o cabecalho so e
    # escrito entao). Pular os ilegiveis em vez de quebrar o monitor.
    # Correcao D-13: um CSV ilegivel era descartado em silencio e o monitor
    # reportava progresso pela metade. Numa run de 25 h o operador ve "4000/7920" e
    # conclui que esta na metade, quando uma corrente inteira sumiu do relatorio.
    partes, ilegiveis = [], []
    for f in fms:
        try:
            if f.stat().st_size > 0:
                # `carimbo` FORCADO a texto. Um hash de 10 digitos hexadecimais
                # pode ter a forma de notacao cientifica — o da gemea de banda,
                # `8003e33729`, e lido como `inf` — e dois carimbos distintos
                # colapsariam no MESMO valor, fazendo a checagem de mistura passar
                # exatamente no caso que ela existe para pegar.
                partes.append(pd.read_csv(f, dtype={"carimbo": str}))
        except Exception as e:
            ilegiveis.append((f, type(e).__name__))
    if ilegiveis:
        print(f"[{agora}] ATENCAO: {len(ilegiveis)} CSV(s) ilegiveis — o progresso "
              f"abaixo esta INCOMPLETO:")
        for f, tipo in ilegiveis:
            print(f"    {f} ({tipo})")
    if not partes:
        print(f"[{agora}] CSVs ainda vazios — a primeira celula nao fechou")
        return 0
    fm = max(fms, key=lambda f: f.stat().st_mtime)
    m = pd.concat(partes, ignore_index=True)
    if m.empty:
        print(f"[{agora}] CSV vazio"); return 0
    m["montagem"] = m["montagem"].astype(str)

    # Correcao B-10: o agregador se recusa a montar uma tabela que misture
    # contratos. O monitor concatena CSVs de diretorios diferentes (as duas
    # correntes, mais o que ja estava la), e sem esta checagem uma run feita antes
    # de uma correcao e outra feita depois somam sem erro nenhum — que e exatamente
    # como as grades invalidadas se formaram. Aqui o aviso e visivel e o total nao
    # e calculado sobre a mistura.
    selos = sorted(set(m["carimbo"].dropna().astype(str))) if "carimbo" in m.columns else []
    sem_selo = int(m["carimbo"].isna().sum()) if "carimbo" in m.columns else len(m)
    if len(selos) > 1 or (selos and sem_selo):
        print(f"[{agora}] RECUSA: a tabela misturaria contratos diferentes")
        for s in selos:
            print(f"    {s}: {int((m['carimbo'].astype(str) == s).sum())} linhas")
        if sem_selo:
            print(f"    sem carimbo (anterior ao contrato): {sem_selo} linhas")
        print("  Separe por diretorio e monitore um contrato por vez.")
        return len(m)

    # Montagens presentes no CSV entram na visao mesmo se nao pedidas, senao o
    # trabalho ja feito fica invisivel (foi o que aconteceu com a fase 0).
    for extra in sorted(m.montagem.unique()):
        if extra not in mts:
            mts = mts + [extra]

    # Ritmo pelo RELOGIO DE PAREDE, nao pela coluna secs: ela cronometra so o
    # fold e ignora o treino do pool, que e o custo dominante (200+ s por alvo).
    span_min = max((time.time() - min(f.stat().st_ctime for f in fms)) / 60, 1e-6)
    cel_por_min_global = len(m) / span_min

    ritmo, feitos = {}, {}
    for (mt, md), g in m.groupby(["montagem", "modelo"]):
        feitos[(mt, md)] = g.target.nunique()

    # min/alvo estimado: escala o ritmo global observado pelo numero de canais,
    # relativo a media de canais ja processada. E grosseiro no inicio e converge.
    ch_feitos = [N_CANAIS.get(str(x), 8) for x in m.montagem]
    ch_medio = (sum(ch_feitos) / len(ch_feitos)) if ch_feitos else 8.0
    min_por_alvo_medio = (A.folds / cel_por_min_global) if cel_por_min_global > 0 else 0.0

    def est(mt, md):
        return min_por_alvo_medio * N_CANAIS.get(mt, 8) / max(ch_medio, 1)

    print(f"[{agora}]  benchmark por montagem")
    print(f"{'montagem':<10}{'modelo':<17}{'alvos':>7}{'%':>6}{'min/alvo':>10}{'restam':>9}")
    print("-" * 60)
    restante, feito_cel = 0.0, 0
    for mt in mts:
        for md in mds:
            na = feitos.get((mt, md), 0)
            falta = max(A.alvos - na, 0)
            e = est(mt, md)
            restante += falta * e
            feito_cel += len(m[(m.montagem == mt) & (m.modelo == md)])
            marca = "" if (mt, md) in ritmo else "~"
            print(f"{mt:<10}{md:<17}{na:>4}/{A.alvos:<2}{100*na/A.alvos:>5.0f}%"
                  f"{e:>9.1f}{marca}{falta*e/60:>8.1f}h")
    total = len(mts) * len(mds) * A.alvos * A.folds
    print("-" * 60)
    print(f"  celulas: {feito_cel}/{total} ({100*feito_cel/total:.1f}%)   ~ = ritmo extrapolado")
    fim = datetime.now() + timedelta(minutes=restante)
    print(f"  RESTAM ~{restante/60:.1f} h  ->  termino ~{fim.strftime('%d/%m %H:%M')}")

    n = len(m)
    if anterior is not None:
        if n == anterior:
            idade = (time.time() - fm.stat().st_mtime) / 60
            print(f"  ATENCAO: sem celulas novas desde a ultima leitura "
                  f"(CSV parado ha {idade:.0f} min) — a run pode ter terminado ou caido")
        else:
            print(f"  +{n - anterior} celulas desde a ultima leitura")

    fps = sorted(csvs_ativos(Path(A.dir), "predicoes.csv"))
    if fps:
        n_p = sum(sum(1 for _ in open(f, encoding="utf-8")) - 1 for f in fps)
        print(f"  predicoes por epoca: {n_p} linhas em {len(fps)} arquivos")

    if A.log and Path(A.log).exists():
        txt = Path(A.log).read_text(encoding="utf-8", errors="replace")
        erros = re.findall(r"^\s*ERRO em .*$", txt, re.M)
        if erros:
            print(f"  erros: {len(erros)}")
            for e in erros[-3:]:
                print(f"    {e.strip()[:96]}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(EXP / "results" / "bench_montagens"))
    ap.add_argument("--montagens", nargs="+", default=["todas"])
    ap.add_argument("--modelos", nargs="+", default=["todos"])
    ap.add_argument("--alvos", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--log", default=None)
    ap.add_argument("--watch", action="store_true",
                    help="reimprime periodicamente ate Ctrl+C")
    ap.add_argument("--intervalo", type=int, default=300,
                    help="segundos entre atualizacoes no modo --watch (padrao 300)")
    A = ap.parse_args()

    if not A.watch:
        relatorio(A, None)
        return

    anterior = None
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"--watch a cada {A.intervalo}s — Ctrl+C para sair\n")
            anterior = relatorio(A, anterior)
            time.sleep(A.intervalo)
    except KeyboardInterrupt:
        print("\nmonitor encerrado (a run continua)")


if __name__ == "__main__":
    main()
