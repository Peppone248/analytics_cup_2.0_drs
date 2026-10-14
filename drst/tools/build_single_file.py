"""Genera drst.py single-file dal package. Rimuove gli import relativi
OVUNQUE, anche dentro i corpi delle funzioni (bug risolto 11 ott)."""
import re, pathlib, ast, sys

ORDER=["config","priors","stress","indexing","curve","aggregate","validate",
       "tracking_features","pipeline","plotting","animate","evaluate_upgrade"]
TITLES={"config":"CONFIGURAZIONE","priors":"PRIOR DI PERICOLOSITA' (geometrico, spaziale, subtype)",
 "stress":"STRESS per corsa — containment + vulnerabilita' strutturale",
 "indexing":"FRAME-BASED INDEXING STRATEGY","curve":"CURVA DRS(t)",
 "aggregate":"AGGREGAZIONE per squadra e difensore","validate":"VALIDAZIONE",
 "tracking_features":"FEATURE DAL TRACKING XY — curva sep(t) e struttura",
 "pipeline":"PIPELINE end-to-end","plotting":"FIGURE statiche",
 "animate":"ANIMAZIONE 2D su campo","evaluate_upgrade":"CONFRONTO fra configurazioni"}
DROP={"numpy","pandas","abc","dataclasses","pathlib","typing","__future__","json","collections"}

def strip_imports(src: str) -> str:
    """Rimuove import relativi (level>0) e import gia' presenti nell'header,
    a QUALUNQUE profondita'. Se togliendoli il corpo di una funzione resta
    vuoto, inserisce `pass` con l'indentazione giusta."""
    tree = ast.parse(src)
    kill = []
    for node in ast.walk(tree):                      # <-- walk, non tree.body
        if isinstance(node, ast.ImportFrom):
            if node.level > 0 or (node.module or "").split(".")[0] in DROP:
                kill.append((node.lineno, node.end_lineno, node.col_offset))
        elif isinstance(node, ast.Import):
            if all(a.name.split(".")[0] in DROP for a in node.names):
                kill.append((node.lineno, node.end_lineno, node.col_offset))

    lines = src.split("\n")
    drop = {i for a, b, _ in kill for i in range(a, b + 1)}

    # i blocchi (funzioni, if, for...) che restano senza corpo prendono `pass`
    keep = []
    for n, line in enumerate(lines, 1):
        if n in drop:
            continue
        keep.append((n, line))
    out = [l for _, l in keep]
    text = "\n".join(out)
    try:
        ast.parse(text)
    except (SyntaxError, IndentationError):
        # reinserisci un `pass` dove un blocco e' rimasto vuoto
        out = []
        for n, line in enumerate(lines, 1):
            if n in drop:
                indent = next((c for a, b, c in kill if a <= n <= b), 0)
                nxt = lines[n] if n < len(lines) else ""
                cur_ind = len(line) - len(line.lstrip())
                nxt_ind = len(nxt) - len(nxt.lstrip()) if nxt.strip() else -1
                if nxt_ind >= 0 and nxt_ind < cur_ind:
                    out.append(" " * cur_ind + "pass")
                continue
            out.append(line)
        text = "\n".join(out)
    return text

HEADER_TMPL = '''"""
drst.py — DRS(t): Defensive Response Signature come curva temporale.
Versione single-file. Equivalente al package drst/, nessuna installazione.

    stress = danger_prior             (geometrico: exp(-dist/25m), non usa
                                       gli esiti; oppure da subtype/zona)
           x containment_failure      (separation gain)
           x structural_vulnerability (enriched, + tracking XY se c'e')

Lo stress NON usa mai gli esiti (targeted/received/lead_to_shot): restano
fuori per la validazione.

USO IN VS CODE
    1. metti questo file accanto ai dati
    2. imposta i path nel blocco CONFIGURA QUI
    3. F5 (o: python drst.py)

Da notebook o altro script nella stessa cartella:
    from drst import run_pipeline, extract_all, animate_match, compare

FLUSSO COMPLETO
    1. feature di tracking, una volta sola (~4 s per match):
         extract_all("data/layer3_enriched.csv",
                     "E:/Projects/opendata/data/matches",
                     "data/tracking_features.csv")
    2. imposta TRACKING_FEATURES qui sotto e rilancia
    3. per confrontare le configurazioni:
         compare("data/layer3_enriched.csv", "data/tracking_features.csv",
                 "out/upgrade")

DIPENDENZE
    obbligatorie  pandas numpy pyarrow
    validazione   statsmodels scikit-learn
    figure        matplotlib    (+ ffmpeg per gli MP4, altrimenti GIF)
    pip install pandas numpy pyarrow statsmodels scikit-learn matplotlib

INDICE (Ctrl+F sui marcatori ===)
{index}    13. MAIN

Risultati di riferimento, limiti noti ed estensioni: vedi README.md
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

# ====== CONFIGURA QUI ======================================================
LAYER3_PATH = Path("data/layer3_enriched.csv")
TRACKING_FEATURES = None      # Path("data/tracking_features.csv"), o None
OUT_DIR = Path("out/drst_v2")
PLOT_MATCH = None             # match_id della figura statica, o None
RUN_OOS = False               # dissociazione out-of-sample (piu' lenta)

PRIOR_KIND = "geometric"      # "geometric" | "spatial" | "subtype"
GEO_SCALE = 25.0              # metri, solo per il prior geometrico
GEO_USE_ENDPOINT = False      # prior su min(partenza, arrivo) invece della
                              # sola partenza. OFF: migliora la singola
                              # corsa (corr +0.403 -> +0.469) e peggiora la
                              # squadra (split-half +0.749 -> +0.628).
                              # Accenderlo per animazioni e case study,
                              # non per le metriche di squadra. Vedi README
GEO_BALL_GATE = False         # OFF: attenua il prior se la palla e' lontana,
                              # ma aggiunge +0.001 di correlazione = rumore
STRUCTURAL = None             # None = auto
USE_TRACKING_STRUCTURE = True # str_stretch_rate e str_recovery_rate
USE_CURVE_CONTAINMENT = False # OFF: sui 20 match peggiora (vedi README)
CONTAINMENT_KIND = "separation_gain"
                              # "separation_gain" | "learned". Il learned e'
                              # il residuo rispetto a un'attesa appresa dalla
                              # geometria: porta la split-half del containment
                              # da +0.026 a +0.605. Richiede TRACKING_FEATURES
                              # e scikit-learn. Resta OFF di default perche'
                              # dentro il prodotto dello stress peggiora la
                              # metrica standardizzata per fase: va letto
                              # come asse separato in team_two_axis.csv,
                              # che lo usa comunque. Vedi README

# --- animazione 2D (opzionale)
ANIMATE_MATCH = None          # match_id, o None per saltare
TRACKING_PATH = Path("data/tracking.jsonl")
N_WINDOWS = 5
SPATIAL_MODE = "gap"          # "runner" | "projected" | "gap"
FPS = 10
# ===========================================================================

'''

MAIN = '''
# ===========================================================================
# 13. MAIN
# ===========================================================================

def main():
    cfg = replace(DEFAULT, prior_kind=PRIOR_KIND, geo_prior_scale=GEO_SCALE,
                  geo_use_endpoint=GEO_USE_ENDPOINT,
                  geo_ball_gate=GEO_BALL_GATE,
                  use_tracking_structure=USE_TRACKING_STRUCTURE,
                  use_curve_containment=USE_CURVE_CONTAINMENT,
                  containment_kind=CONTAINMENT_KIND)

    layer3 = LAYER3_PATH
    if TRACKING_FEATURES is not None:
        merged = merge_into_layer3(load_layer3(LAYER3_PATH, cfg),
                                   pd.read_csv(TRACKING_FEATURES), cfg)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        layer3 = OUT_DIR / "_layer3_tracking.csv"
        merged.to_csv(layer3, index=False)
        cov = merged["sep_min"].notna().mean() if "sep_min" in merged else 0.0
        print(f"Feature di tracking unite: copertura {cov:.1%}")
        if GEO_USE_ENDPOINT:
            gc = (merged["geo_dist_min"].notna().mean()
                  if "geo_dist_min" in merged else 0.0)
            if gc == 0.0:
                print("[!] geo_dist_min assente: il prior usa la sola "
                      "partenza. Ri-estrarre le feature di tracking con la "
                      "versione aggiornata per attivare il punto di arrivo.")
            else:
                print(f"Punto di arrivo su {gc:.1%} delle corse "
                      f"(avvicinamento medio alla porta "
                      f"{merged['geo_progress'].mean():+.1f} m)")

    res = run_pipeline(layer3, cfg=cfg, structural=STRUCTURAL, with_oos=RUN_OOS)
    out = res.save(OUT_DIR)
    if res.two_axis is not None:
        print("\\nProfilo a due assi (alto = peggio su entrambi gli assi)")
        cols = ["defending_team_id", "n_matches", "n_runs", "esposizione",
                "cedimento", "deep_phase_share"]
        print(res.two_axis[cols].round(3).to_string(index=False))
    print(f"\\nSalvato in {out.resolve()}")

    if PLOT_MATCH is not None:
        try:
            import matplotlib
        except ImportError:
            print("[!] matplotlib non installato: figura saltata.")
        else:
            matplotlib.use("Agg")
            curve, _ = res.curves_by_match[PLOT_MATCH]
            runs = res.runs[res.runs[cfg.col_match] == PLOT_MATCH]
            fig = plot_match_curve(curve, runs, cfg, title=f"DRS(t) - match {PLOT_MATCH}")
            p = out / f"curve_{PLOT_MATCH}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight")
            print(f"Figura: {p.resolve()}")

    if ANIMATE_MATCH is not None:
        animate_match(ANIMATE_MATCH, out / "runs_with_stress.parquet",
                      out / "curves.npz", TRACKING_PATH, out / "clips",
                      n_windows=N_WINDOWS, cfg=cfg,
                      spatial_mode=SPATIAL_MODE, fps=FPS)
    return res


if __name__ == "__main__":
    main()
'''

def build(pkg_dir: pathlib.Path, out_path: pathlib.Path) -> str:
    index = "".join(f"    {i:2d}. {TITLES[m]}\n" for i, m in enumerate(ORDER, 1))
    parts = [HEADER_TMPL.format(index=index)]
    for i, m in enumerate(ORDER, 1):
        src = (pkg_dir / f"{m}.py").read_text()
        src = re.sub(r'^"""(?:.|\n)*?"""\n', "", src, count=1)
        src = re.split(r'\nif __name__ == "__main__":', src)[0]
        parts.append(f"\n# {'='*75}\n# {i}. {TITLES[m]}\n# {'='*75}\n\n"
                     + strip_imports(src).strip("\n") + "\n")
    parts.append(MAIN)
    txt = "".join(parts)
    for a, b in [("V.dissociation_oos", "dissociation_oos"), ("V.dissociation(", "dissociation("),
                 ("V.split_half", "split_half"), ("V.pre_run_drs", "pre_run_drs")]:
        txt = txt.replace(a, b)
    ast.parse(txt)                      # fallisce subito se la sintesi e' rotta
    out_path.write_text(txt)
    return txt

def build_tests(tests_src: pathlib.Path, out_path: pathlib.Path) -> str:
    """Adatta la suite del package al file unico.

    Nel single-file tutto sta in drst.py, che e' un MODULO e non un package:
    `from drst.priors import ...` fallisce con "'drst' is not a package".
    Qui le import annidate vengono appiattite su `from drst import ...`.
    """
    src = tests_src.read_text()
    # ATTENZIONE: anche gli import ANNIDATI nelle funzioni, non solo quelli
    # in testa. E' lo stesso punto cieco che aveva strip_imports prima di
    # passare ad ast.walk: due test importano dentro il corpo.
    src = re.sub(r"^(\s*)from drst\.[\w.]+ import ", r"\1from drst import ",
                 src, flags=re.M)
    # NESSUN dedup: un import annidato identico a uno in testa serve
    # comunque nel suo scope, e rimuoverlo da' NameError. Righe duplicate
    # sono innocue.
    src = src.replace('"""Test del package drst. pytest tests/"""',
                      '"""Test del file unico drst.py.  pytest test_drst.py"""')
    ast.parse(src)
    out_path.write_text(src)
    return src


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    t = build(pathlib.Path("drst"), out)
    print(f"righe: {t.count(chr(10))}")
    ts = build_tests(pathlib.Path("tests/test_drst.py"),
                     out.parent / "test_drst.py")
    print(f"test: {ts.count(chr(10))} righe")
