# drsframe — Defensive Response, frame by frame

**Dagli una partita, ottieni la curva di risposta difensiva** — e i video per lo staff.

Una metrica di valutazione dell'azione difensiva costruita **frame per frame** dal
tracking SkillCorner (open data, 20 match A-League 24/25), che misura quanto una difesa
concede **rispetto a quanto concederebbe una difesa media contro lo stesso attacco**, pesa
ogni momento con la pericolosità dell'attacco subito (z-score sugli eventi) e produce un
ranking interno a ogni partita per episodi, squadre e difensori, più i video 2D.

> **v0.2**: risposta divisa in due canali (opzioni coperte / pressione sul portatore),
> placebo, risultati partita per partita, carico contro risposta, migliore risposta
> osservata (analoghi) e video per lo staff. Dettagli e numeri in `CHANGELOG.md`.

> Estensione naturale del `DenseTrackingIndex` previsto in `drst`: lì il DRS(t) è
> ancorato alle corse (40% dei frame coperti), qui è definito su **ogni frame** in cui una
> squadra è in possesso (63% della partita, il resto sono stoppage).

---

## L'idea in tre righe

```
X(t)    = valore lasciato scoperto dalla difesa adesso         (geometria + tracking)
X̂(t)    = quanto ne lascerebbe una difesa media contro questo attacco   (modello OOF)
R(t)    = X̂(t) − X(t)   > 0  la difesa fa meglio dell'atteso
```

Il punto chiave è che **X̂ vede solo l'attacco** (palla, posizioni e velocità degli
attaccanti, mai i difensori): il residuo R è la parte di esposizione attribuibile alla
difesa. È l'idea del *ghosting* (Le et al. 2017) in forma statistica: invece di simulare un
difensore medio, si stima l'esposizione media che il campionato concede in quella
configurazione d'attacco.

## Pipeline

```
tracking JSONL ─► orientamento (attacco sempre verso +x) ─► velocità lisciate
      │
      ├─► time-to-intercept per ogni coppia difensore → punto     (Spearman 2018, Bekkers 2025)
      │      • copertura dell'opzione j: arrivo sul punto di ricezione O taglio della linea di passaggio
      │      • pressione sul portatore (Pressing Intensity, soglia 1.5 s)
      │
      ├─► marcatura: assegnazione ungherese difensore → attaccante, costo = distanza dal
      │      punto ideale lato porta, opzione "a zona" oltre 12 m, isteresi 2 m
      │
      ├─► valore della posizione V(x,y): Nadaraya-Watson sull'EPV SkillCorner dei possessi
      │
      ├─► X(t) esposizione, credit/blame per difensore (leave-one-out esatto)
      │
      ├─► X̂(t) gradient boosting Poisson, GroupKFold 5 fold su match_id
      │
      ├─► attacchi subiti: player_possession (palla al piede) + off_ball_run (senza palla)
      │      pericolo → log → z-score per tipo di evento sul campionato
      │
      └─► ranking interno alle partite · validazione · video 2D
```

### 1. Geometria (frame per frame)

Per ogni difensore *i* e ogni punto *q*:

```
T_i(q) = t_r + | p_i + v_i·t_r − q | / v_max          t_r = 0.7 s, v_max = 7 m/s
p      = 1 / (1 + exp(−π/(√3·σ) · (t_palla − T_i(q))))  σ = 0.45 s
```

- **Copertura dell'opzione j** (attaccante senza palla): il difensore la contende se arriva
  sul punto di ricezione proiettato `q_j` entro l'arrivo della palla (+0.5 s) **oppure** se
  taglia la linea palla→`q_j` prima della palla (8 punti campionati).
  `C_j = 1 − Π_i (1 − p_ij)`.
- **Pressione sul portatore**: stessa logistica con soglia 1.5 s come in Pressing Intensity.
- **Distanze richieste**: per ogni difensore e frame `d_ball` (dalla palla), `d_mark`
  (dall'uomo assegnato), `d_ideal` (dal punto di marcatura ideale, 1.5 m dall'attaccante
  verso la porta), `goal_side` (è più vicino alla porta del suo uomo?).

Sanity check sul match 1996436: assegnazione trovata per l'81% dei frame-difensore,
`d_mark` mediana 5.1 m, lato porta 82% dei frame.

### 2. Esposizione

```
gain_j  = max(0, V(q_j) − V(palla))          minaccia che l'opzione aggiunge (logica xT)
x_j     = gain_j · (1 − C_j)                 quanta resta scoperta
x_car   = (V(palla) − V_min) · (1 − Press)   portatore libero di progredire
X(t)    = Σ_j x_j + x_car                    unità: EPV (prob. di gol del possesso)
```

I passaggi all'indietro valgono zero: una difesa non viene penalizzata per lasciare libero
il retropassaggio.

**Attribuzione individuale** (stessa domanda della LOO nella tesi di Filippo, ma geometrica
e in forma chiusa, quindi calcolabile su ogni frame):

- `credit_i` = di quanto salirebbe X togliendo il difensore *i* (prodotti prefisso/suffisso,
  esatto anche quando `p_ij → 1`; verificato contro il ricalcolo brute force nei test);
- `blame_i` = esposizione residua dell'attaccante che *i* dovrebbe marcare.

### 3. Esposizione attesa (KFold)

`HistGradientBoostingRegressor(loss="poisson")`, 18 feature solo d'attacco, allenato su
un frame ogni 5 (2 Hz) e predetto out-of-fold con **GroupKFold su match_id**: nessuna
partita contribuisce alla propria attesa (c'è un test che lo verifica).

| fold | R² | Spearman |
|---:|---:|---:|
| 0 | 0.467 | 0.736 |
| 1 | 0.397 | 0.717 |
| 2 | 0.514 | 0.732 |
| 3 | 0.461 | 0.732 |
| 4 | 0.470 | 0.729 |

La fase SkillCorner (`team_in_possession_phase_type`) è **esclusa** dal contesto: è
etichettata a posteriori (una fase diventa *finish* anche perché la difesa ha lasciato
arrivare l'attacco), quindi assorbirebbe parte dell'effetto difensivo. Toglierla non cambia
l'R² (0.46) e alza la stabilità per squadra (0.58 → 0.61).

### 4. Pericolosità degli attacchi in z-score

| famiglia | evento | pericolo grezzo |
|---|---|---|
| palla al piede | `player_possession` | `possession_epv_for_start` |
| lontano dalla palla | `off_ball_run` | `xthreat` |

```
z_e = (log(raw_e + 1e-4) − μ_tipo) / σ_tipo        peso w_e = Φ(z_e) ∈ (0, 1)
```

Standardizzare per tipo rende confrontabili le due scale. Per ogni evento si misura R medio
sulla finestra `[inizio − 1 s, fine]` (`R_win`) e, come controllo non meccanico, sui 2 s
**prima** che l'attacco parta (`R_pre`, non può contenere il tiro).

### 5. Ranking interno alle partite

- **Episodi** — `rank_danger_in_match` (1 = attacco più pericoloso subito da quella squadra
  in quella partita), `rank_response_in_match` (1 = difeso peggio), `critical` = z > 1 e R < 0
  (6% degli attacchi: sono quelli che finiscono nel video).
- **Squadre** — `DRF = Σ w_e·R_e / Σ w_e`: risposta difensiva pesata per pericolo. Le due
  squadre di ogni partita sono ordinate, poi tabella di campionato con shrinkage
  empirico-bayes (Auckland ha 7 partite, Newcastle 1).
- **Difensori** — credit, blame, `net = credit − blame` per 90' fuori possesso.
  Il ranking usa lo scarto dalla mediana del **proprio gruppo di ruolo** (CB, FB, MF, WM, FW):
  il credit grezzo favorisce gli esterni, che coprono più opzioni dei centrali.

---

## Risultati (20 match, 620.378 frame, 27.982 attacchi)

![DRF per squadra](docs/fig1_league_drf.png)

### Cosa regge

**1. La geometria della copertura è credibile.** I runner che il modello vede scoperti
vengono serviti di più:

| esito | AUROC apertura max | tasso Q1 apertura | tasso Q5 apertura |
|---|---:|---:|---:|
| `received` | **0.724** | 17.3% | 40.2% |
| `targeted` | 0.644 | 32.9% | 41.8% |

**2. A parità di pericolo, la risposta difensiva conta.** Logistica
`lead_to_shot ~ danger_z + R` per attacco, SE cluster per partita:

| finestra | evento | n | OR pericolo (+1 z) | **OR risposta (+1 sd)** | p |
|---|---|---:|---:|---:|---:|
| durante l'attacco | `off_ball_run` | 9.897 | 3.94 | **0.83** | < 1e-4 |
| durante l'attacco | `player_possession` | 18.084 | 3.30 | **0.88** | < 1e-4 |
| 2 s prima che parta | `off_ball_run` | 9.869 | 3.94 | **0.86** | 0.001 |
| 2 s prima che parta | `player_possession` | 18.040 | 3.31 | **0.93** | 0.02 |

La versione "prima che parta" non può contenere il tiro, quindi l'effetto non è meccanico.
Dalla figura sotto: l'effetto è concentrato nel quintile di risposta peggiore, sugli
attacchi pericolosi (z ≥ 1: 40% di tiri contro 27–36% negli altri quintili).

![Effetto della risposta](docs/fig2_response_effect.png)

**3. È un tratto di squadra.** Split-half (12 squadre con ≥ 2 partite, 500 divisioni casuali
delle partite):

| metrica | mediana | IQR | % positivi |
|---|---:|---:|---:|
| `DRF` | **0.613** | 0.53 – 0.72 | 100% |
| `R_frame_mean` | 0.628 | 0.52 – 0.75 | 99.6% |
| `n_attacks` (volume, riferimento) | 0.402 | 0.26 – 0.54 | 94% |

Con etichette di squadra permutate la mediana è −0.01 (5°–95° percentile −0.39 / +0.51):
0.61 è sopra il 95° percentile del nullo. Residualizzando il DRF su volume d'attacchi e X̂
medio la stabilità scende a **0.34** (97% positivi): metà della stabilità passa per
*quanto* e *come* una squadra viene attaccata, l'altra metà è difesa.

**4. Anche i difensori.** Split-half su 138 giocatori con ≥ 2 partite:
`d_mark` 0.64, `credit_p90` 0.48, `blame_p90` 0.46, `net_vs_role` 0.35.

**5. La risposta ha due canali che fanno lavori diversi.** L'esposizione si divide in
opzioni di passaggio scoperte (`X_opt`) e spazio al portatore (`X_car`); ciascuna ha la sua
attesa e il suo residuo. I due canali sono ortogonali (r = 0.01). Misura nei 2 s prima
dell'evento, logistica con entrambi:

| attacco | esito | OR canale opzioni | OR canale portatore |
|---|---|---:|---:|
| corsa | `targeted` / `received` | 0.99 / 0.98 (ns) | 1.03 / 1.02 (ns) |
| corsa | tiro | **0.90** (p 0.002) | 0.91 (p 0.07) |
| possesso | tiro | **0.94** (p 0.03) | 0.95 (ns) |
| possesso | perdita palla | 1.01 (ns) | **1.24** (p 6e-9) |

Coprire le opzioni evita i tiri, pressare il portatore fa recuperare palla. Per squadra la
copertura delle opzioni (`DRF_opt`) e' il tratto piu' stabile misurato: split-half **0.73**.

**6. Placebo: serve la difesa di quel momento.** Sostituendo le posizioni dei difensori con
le loro di 90 s o 300 s prima (attacco e palla veri) e ricalcolando tutto, ogni effetto
torna a ~1:

| test | reale | placebo 90 s | placebo 300 s |
|---|---:|---:|---:|
| corsa → tiro, R prima della corsa | **0.86** (p 0.001) | 1.00 (p 0.99) | 1.01 (p 0.91) |
| possesso → tiro, R prima | **0.93** (p 0.02) | 1.01 (p 0.77) | 1.05 (p 0.15) |
| corsa → tiro, canale opzioni | **0.90** (p 0.002) | 1.01 (p 0.91) | 1.01 (p 0.93) |
| possesso → perdita, canale portatore | **1.24** (p 6e-9) | 1.01 (p 0.79) | 1.03 (p 0.05) |

**7. Partita per partita (leave-one-match-out).** Modello stimato su 19 partite, valutato
sulla ventesima:

| test | log-loss migliora | AUROC migliora |
|---|---:|---:|
| corsa → tiro, + R | **14/20** | 13/20 |
| corsa → tiro, + due canali | 13/20 | **15/20** |
| corsa → ricevuta, + R | 1/20 | 7/20 |
| possesso → perdita palla, + canale portatore | 13/20 | **16/20** |

La dissociazione regge anche qui: la risposta non predice se la corsa viene servita (1/20).

**8. Quanto vieni attaccato e come rispondi sono due cose diverse.** Il carico subito
(stress per corsa di `drst`) e la risposta (DRF) sono quasi indipendenti; i tiri concessi,
la metrica che si usa oggi, seguono il carico (Spearman +0.71 squadra-partita), non la
risposta (−0.23).

![Carico contro risposta](docs/fig3_load_vs_response.png)

**9. La migliore risposta osservata spiega i tiri meglio dell'attesa del modello.** Per ogni
attacco si cercano gli analoghi (stesso tipo e subtype, configurazione d'attacco simile,
stesso pericolo atteso, altra partita e altra squadra) e si confronta lo spazio concesso con
quello concesso da loro. AIC corse 3531 → 3516, possessi 2905 → 2890; nello stesso modello
l'effetto passa al confronto con gli analoghi (OR 0.74 e 0.61). Supporto: il 61% delle corse
critiche ha almeno 5 analoghi; solo il 21% dei possessi critici.

### Cosa non regge (e va detto)

**Frame per frame la difesa non aggiunge informazione sul tiro.** Tiro della squadra in
possesso entro 5 s, gradient boosting OOF:

| modello | AUROC | log-loss |
|---|---:|---:|
| solo contesto d'attacco | 0.9335 | 0.10063 |
| contesto + geometria difensiva | 0.9340 | 0.10053 |
| Δ (bootstrap per partita, IC 95%) | +0.0005 [−0.0007, +0.0017] | +0.0001 [−0.0008, +0.0008] |

È lo stesso risultato nullo della RQ3 della tesi di Filippo e del `p(targeted)` di `drst`:
a scala di frame la posizione degli attaccanti satura la previsione. Il segnale difensivo
emerge **aggregando per attacco** (punto 2) e **per squadra** (punto 3), non nel frame
singolo. Il paper dovrebbe dirlo esplicitamente.

**Sui possessi la risposta non generalizza al tiro partita per partita** (9/20 in
leave-one-match-out): l'OR pooled 0.88 e' trainato da poche partite. Sui possessi conta il
canale portatore, ma per il recupero palla, non per il tiro.

**A livello di partita il DRF non predice i tiri concessi** (Poisson con controllo per numero
e pericolo degli attacchi: p = 0.58). Il pericolo medio degli attacchi subiti spiega quasi
tutto (Spearman 0.82 con i tiri).

### Esempio di report di partita — 1996436, Auckland FC vs Macarthur FC

| difende | attacchi | pericolosi (z>1) | DRF (mEPV) | tiri concessi | rank |
|---|---:|---:|---:|---:|---:|
| Auckland FC | 545 | 84 | +0.26 | 12 | 1 |
| Macarthur FC | 638 | 98 | −0.04 | 20 | 2 |

---

## Video 2D

Due stili dalla stessa pipeline.

**Per lo staff** (default):

```bash
python -m drsframe.cli video --data opendata/data/matches --cache cache --out out/run \
    --match 1996436 --n 3 --style staff --lang it      # oppure --lang en
```

Per ogni episodio critico: scheda con il verdetto in una frase → clip con rallentatore
sull'azione e fermo immagine nel momento di massimo spazio concesso → come ha difeso lo
stesso attacco la squadra con la miglior risposta osservata → scheda di confronto. Una sola
misura, in parole: **spazio concesso come multiplo del normale** (1× = quanto concede una
difesa media del campionato in quella situazione), piu' i due canali con lo stesso
linguaggio. Sul fermo immagine al massimo tre frasi automatiche ("#44 nessuno lo marca ·
il piu' vicino #17 a 12 m"). Escono anche il fermo immagine PNG per le slide e un indice CSV.
L'attacco va sempre verso destra.

![Fermo immagine del video staff](docs/staff_keyframe_example.png)

**Per analisti** (`--style analyst --montage`): tutte le grandezze del modello (X, X̂, R,
esposizione per opzione, colori della marcatura per distanza dal punto ideale).

## Uso

```bash
pip install -e ".[fast,test]"

# dati: clone dell'opendata + tracking (git lfs pull, oppure media.githubusercontent.com)
python -m drsframe.cli run --data opendata/data/matches --cache cache --out out/run
```

```python
from drsframe import run_pipeline, Config
res = run_pipeline("opendata/data/matches", "cache", Config())
res.team_match          # ranking delle due squadre in ogni partita
res.attacks             # ogni attacco subito: z-score, R e canali, rank nel match, marcatore, analoghi
res.defender_rank       # difensori per partita
res.frames              # 620k frame: X, X̂, R e componenti
```

Riproduzione completa da zero (dati, test, pipeline, placebo, figure, video): `bash reproduce.sh`.

Costi: ~12 s a partita per la geometria (poi in cache `.npz`), ~20 min la pipeline completa
con validazione su 2 core, ~25 min il placebo, ~3 min a episodio per il video staff.

| file di output | contenuto |
|---|---|
| `frames.parquet` | una riga per frame: X, X̂, R, componenti, contesto |
| `attacks_ranked.csv` | attacchi subiti con z-score, risposta, rank nel match, marcatore |
| `team_match_scores.csv` | DRF e ranking delle squadre in ogni partita |
| `team_league_table.csv` | tabella di campionato con shrinkage EB |
| `defender_ranking.csv` | difensori per partita, aggiustati per ruolo |
| `validation_*.csv`, `validation.json` | tutte le tabelle di questa sezione |
| `placebo_response.csv`, `placebo_channels.csv` | reale contro placebo a 90 s e 300 s |
| `staff/{match}_staff.mp4`, `{match}_epK_keyframe.png`, `{match}_index.csv` | video staff, fermi immagine, indice |

## Architettura

```
drsframe/
├── config.py      ogni parametro è una scelta di modellazione
├── io.py          JSONL → array (T, N, 2); team/ruoli da match.json; fasi per frame
├── geometry.py    orientamento, velocità, time-to-intercept, copertura, marcatura
├── value.py       superficie V(x, y) dall'EPV SkillCorner
├── exposure.py    X(t), componenti, credit/blame (LOO esatto), feature di contesto
├── expected.py    X̂(t) out-of-fold con GroupKFold, totale e per canale
├── events.py      attacchi subiti, z-score, risposta per evento, geometria del marcatore
├── rank.py        ranking interno alle partite: episodi, squadre, difensori
├── validate.py    test sugli esiti, canali, leave-one-match-out, split-half, analoghi
├── placebo.py     difensori di un altro momento
├── analogs.py     migliore risposta osservata (analoghi in altre partite)
├── load.py        asse del carico (stress di drst, fallback X̂)
├── build.py       loop sui match con cache
├── pipeline.py    run_pipeline()
├── render.py      video 2D per analisti
├── staff_video.py video per lo staff: frasi, fermo immagine, confronto con l'analogo
├── highlights.py  scelta degli episodi e montaggio
├── figures.py     figure del README
└── cli.py
```

`pytest tests/` — 16 test + 1 xfail. `test_drsframe.py`: logistica, time-to-intercept,
copertura, marcatura, LOO in forma chiusa = ricalcolo brute force, canali che sommano al
totale, superficie di valore, z-score, **predizioni OOF invarianti al target del match
escluso**. `test_scenes.py`: scene costruite a mano (2 contro 1, difensore sulla linea di
passaggio, pressione, retropassaggio, lato porta, isteresi). Lo xfail documenta il limite
del passaggio alto: si sblocchera' da solo quando verra' modellato.

## Limiti

- **Palla a terra.** Copertura e linee di passaggio assumono un passaggio rasoterra a
  15 m/s: cross e lanci (dove si concentrano i tiri da blocco basso) sono modellati male.
  È il limite più probabile dietro al nullo frame-level. Un test su scena lo dimostra: un
  runner alle spalle di una linea piatta risulta coperto (`test_run_behind_flat_line_is_exposed`, xfail).
- **Parametri fissati, non fittati** (t_r, v_max, σ, offset di marcatura): presi da Spearman
  e Pressing Intensity, da sottoporre a sensitività.
- **Marcatura geometrica.** L'ungherese con opzione "a zona" è una baseline; il CDHMM di
  arXiv 2601.00748 la apprende con transizioni dipendenti dal contesto.
- **Campione.** 12 squadre con ≥ 2 partite, Auckland in 7 su 20: lo shrinkage aiuta ma non
  crea dati.
- **Corpo e sguardo assenti.** Il difensore "copre" anche se è girato: è esattamente il buco
  che l'awareness di Filippo (FoV + Kalman) può riempire, entrando come moltiplicatore di
  `p_ij` sui 2 match con body pose.
- **Ruolo di partenza.** I subentrati hanno ruolo `SUB` e finiscono nel gruppo "other".
- **Analoghi sui possessi.** Solo il 21% dei possessi critici ha un riferimento: per quelli il
  video dice "nessun riferimento affidabile" invece di inventarne uno.

## Riferimenti

- Link, Lang, Seidenschwarz (2016), *Real Time Quantification of Dangerousity in Football
  Using Spatiotemporal Tracking Data*, PLOS One — pericolo come zona × controllo × pressione.
- Spearman (2018), *Beyond Expected Goals* / pitch control — tempo di arrivo e logistica.
- Bekkers (2025), *Pressing Intensity: An Intuitive Measure for Pressing in Soccer*,
  arXiv 2501.04712 — time-to-intercept, σ = 0.45, T = 1.5 s.
- Le, Carr, Yue, Lucey (2017), *Data-Driven Ghosting using Deep Imitation Learning*, MIT SSAC.
- *A Machine Learning Framework for Off Ball Defensive Role and Performance Evaluation in
  Football* (2026), arXiv 2601.00748 — marcatura con CDHMM e ghost condizionati al ruolo.
- Teranishi et al. (2025), *Prediction-based evaluation of back-four defense with spatial
  control in soccer*, arXiv 2511.06191.
- Singh (2018), *Introducing Expected Threat (xT)*.
- Karayaz (2026), *collective-press* (github.com/akarayaz/collective-press) — stessa coppa e
  stessi dati: forma del pressing al primo frame, +0.009–0.013 AUC sul solo contesto, nullo del
  body pose fra partite. Lavoro piu' vicino a questo; qui il tempo e' continuo, la risposta e'
  relativa all'attacco e l'oggetto principale sono le corse senza palla.
- Molino, *Defending the Cross: GNN and Situation Awareness for Defensive Evaluation
  in Football*, MSc thesis, Maastricht University.
