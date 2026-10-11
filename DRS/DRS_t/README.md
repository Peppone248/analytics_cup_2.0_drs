# drst — DRS(t)

Defensive Response Signature come **curva temporale** invece che come residuo di un modello sugli esiti.

Ogni off-ball run è uno **stress test** che l'attacco applica alla difesa. Lo stress si accumula e decade: `DRS(t)` è lo stress difensivo istantaneo.

```
stress = danger_prior(subtype) × containment_failure
```

Lo stress **non usa mai gli esiti** (`targeted`, `received`, `lead_to_shot`): restano fuori per la validazione.

---

## Installazione

```bash
pip install -e .
```

Dipendenze: `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `pyarrow`. `matplotlib` solo per le figure.

## Uso

```python
from drst import run_pipeline

res = run_pipeline("data/layer3_enriched.csv")   # v1: attiva tutto quel che l'enriched permette
res.save("out/drst_v1")
```

Il modello ha due livelli, scelti automaticamente in base alle colonne presenti.

| | input | prior | stress |
|---|---|---|---|
| **v0** | `layer3_all_matches.csv` (18 col) | `danger(subtype)` | `prior × containment` |
| **v1** | `layer3_enriched.csv` (47 col) | `danger(subtype, zona)` gerarchico | `prior × containment × vulnerabilità strutturale` |

`run_pipeline` rileva le colonne e attiva la v1 da sola. Per forzare la v0 su dati enriched (utile per l'ablation): `spatial_prior=False, structural=False`, o da CLI `--no-spatial-prior --no-structural`.

Da riga di comando:

```bash
python -m drst.cli --layer3 data/layer3_enriched.csv --out out/drst_v1 --plot-match 1996436
python -m drst.cli --layer3 data/layer3_enriched.csv --out out/drst_oos --oos
```

## Cosa aggiunge la v1

Tre colonne dell'enriched hanno segnale forte e sono tutte osservabili **prima** della corsa (nessun esito).

| colonna | segnale sullo shot rate |
|---|---|
| `delta_to_last_defensive_line_start` | 19.9% (Q1, corsa che parte sulla linea) → 3.2% (Q5). Monotono, 6× |
| `n_simultaneous_runs` | 7.4% con 0 corse simultanee → 37.8% con 4. E il targeted rate **scende** 42.3% → 10.8% |
| `penalty_area_start` | 34.0% in area contro 11.4% fuori (n=265) |

`n_simultaneous_runs` è l'effetto overload misurato direttamente: più corse insieme, ogni singola meno mirata ma la situazione collettiva molto più pericolosa. È il meccanismo del decoy, visibile nei dati.

Le tre si combinano in `structural_vulnerability` ∈ [0,1] con media geometrica pesata (pesi in `DRSConfig`), terzo fattore dello stress.

Il prior spaziale `danger(subtype, third_start)` usa shrinkage gerarchico: ogni cella è tirata verso il prior del proprio subtype, non verso la media globale.

**Nota sul sistema di coordinate.** In `layer3_enriched.csv` le x sono centrate (−52.5/+52.5) e `last_defensive_line_x_start` è **già normalizzato per direzione d'attacco**: le medie coincidono fra `left_to_right` e `right_to_left` (high_block 5.9 vs 6.0). Non va ri-orientato con `attacking_side`. Valori più alti = linea più arretrata (high_block 6, medium 15, low_block 30, set_play 41).

Stratificando per fase, `last_defensive_line_x_start` perde quasi tutto il segnale (medium_block piatto a 3-5%, high_block a 0.5%): è in larga parte un proxy della fase. Per questo non entra nella vulnerabilità strutturale.

## Risultati di riferimento

8877 run, 20 match A-League 24/25, `layer3_all_matches.csv`, config di default.

### Dissociazione — il risultato principale

`DRS(t)` misurato nei 3 secondi **prima** di una corsa, logistica che controlla per subtype (e per zona nella v1):

| esito | v0 in-sample | v0 OOS | v1 in-sample | v1 OOS |
|---|---:|---:|---:|---:|
| `targeted` | 0.987 | 0.988 | 0.988 | 0.986 |
| `received` | 1.028 | 1.029 | 1.024 | 1.018 |
| **`lead_to_shot`** | **1.257** | **1.248** | **1.175** | **1.227** |

Tutti i `p` su `lead_to_shot` sono sotto 1e-9; `targeted` e `received` non sono mai significativi.

L'OR della v1 è più basso perché il test è **più severo**, non perché il modello sia peggiore: usando un prior spaziale la logistica deve controllare anche per la zona, che quindi non può più essere accreditata al `DRS(t)`. È obbligatorio farlo — `dissociation(..., extra_controls=["third_start"])` — altrimenti parte dell'effetto misurato sarebbe il prior stesso.

Il `DRS(t)` non predice se il portatore mira la corsa né se la riceve — quelle sono **scelte dell'attaccante**. Predice fortemente se quella corsa genera un tiro.

È il test che dimostra che il `DRS(t)` misura lo **stato difensivo** e non la scelta dell'avversario: il difetto che aveva affondato il modello di stadio 1 `p(targeted | run)`.

Out-of-sample = prior e normalizzazione per subtype stimati solo sul train fold (GroupKFold su `match_id`).

### Gradiente grezzo

| DRS(t) pre-run | n | shot rate | targeted | received |
|---|---:|---:|---:|---:|
| zero | 2154 | 7.4% | 36.1% | 28.8% |
| Q1 | 1681 | 9.8% | 33.0% | 26.1% |
| Q2 | 1682 | 11.3% | 34.4% | 28.0% |
| Q3 | 1679 | 12.9% | 33.7% | 25.2% |
| Q4 | 1681 | **20.4%** | 33.8% | 25.2% |

Shot rate ×2.75, targeted e received piatti.

Il controllo per subtype tiene su `behind` (1.50×), `support` (1.43×), `run_ahead_of_the_ball` (1.37×), `coming_short` (1.80×). **Cade** su `cross_receiver` (0.99×, pericolo già saturo) e `dropping_off` (0.57×, subtype semanticamente eterogeneo). Sono i due casi in cui la separation da sola non discrimina e serve l'awareness.

### Stabilità per squadra

| metrica | v0 | v1 | % positivi (v1) |
|---|---:|---:|---:|
| `n_runs` (volume) | +0.695 | +0.695 | 99% |
| `drs_mean_all` | +0.525 | +0.581 | 98% |
| `drs_mean_supported` | +0.462 | +0.552 | 85% |
| **`stress_per_run`** | **+0.438** | **+0.627** | 88% |
| `stress_per_run` residualizzato sul volume | — | **+0.416** | 80% |
| `containment_failure_mean` | +0.026 | +0.026 | 52% |

Per confronto, i residui del modello di stadio 1 `p(targeted|run)`: **−0.128**.

Il guadagno della v1 non è volume: `stress_per_run` correla 0.489 con `n_runs`, e residualizzando resta **+0.416**. È il salto da "segnale debole" a "tratto di squadra quasi stabile".

`containment_failure_mean` resta a +0.026 in entrambe le versioni: la v1 non tocca il modello di containment. È la componente che richiede la curva separation(t) o l'awareness.

**Attenzione al volume.** `drs_mean_all` correla 0.83 con `n_runs`: residualizzando sul volume crolla a −0.15. Quella stabilità è quasi tutta "quanto ti attaccano", non "quanto difendi bene". Per i profili di squadra usare solo le metriche normalizzate per volume (`stress_per_run`).

`containment_failure_mean` da solo non è stabile (+0.03): la componente difensiva pura, isolata dal prior di pericolosità, è ancora dentro il rumore. È il sintomo del limite principale — vedi sotto.

### Robustezza del kernel

L'odds ratio su `lead_to_shot` al variare di τ e della forma del kernel:

| τ | rampa | OR |
|---|---|---:|
| 0.5 s | lineare | 1.229 |
| 0.5 s | gradino | 1.239 |
| 2.5 s | lineare | 1.257 |
| 2.5 s | gradino | 1.274 |
| 6.0 s | lineare | 1.236 |
| 6.0 s | gradino | 1.266 |

Il risultato non è un artefatto di tuning.

---

## Architettura

```
drst/
├── config.py      DRSConfig — ogni parametro è una scelta di modellazione
├── priors.py      danger_prior per subtype (EB shrinkage)
├── stress.py      containment_failure + stress per run
├── indexing.py    frame-based indexing strategy ← punto di estensione centrale
├── curve.py       kernel temporale e costruzione DRS(t)
├── aggregate.py   metriche per squadra e per difensore
├── validate.py    dissociazione, split-half, sensibilità al kernel
├── pipeline.py    run_pipeline() end-to-end
├── plotting.py    figure
└── cli.py
```

### Frame-based indexing strategy

L'astrazione su cui poggia l'evoluzione del modello. Separa due cose che altrimenti si confondono:

1. **strategia di indicizzazione** → quali frame esistono, quali hanno supporto, come si pesa un frame senza eventi
2. **deposito dello stress** → come un evento si spalma sui frame

Cambiare (1) senza toccare (2) è il percorso di upgrade dal prototipo event-anchored al `DRS(t)` continuo.

| strategia | stato | cosa fa |
|---|---|---|
| `EventAnchoredIndex` | **attiva** | supporto dove c'è una corsa o la sua coda. Copertura 40% dei frame |
| `DenseTrackingIndex` | stub | fondo geometrico continuo dal tracking XY |
| `PhaseWindowIndex` | stub | unità di analisi = la fase di gioco, non il frame |

`FrameIndex.support` distingue "non misurato" da "difesa sotto controllo". Un aggregato che conta gli zeri dei frame non misurati premia le difese che subiscono poche corse: usare sempre `FrameIndex.masked()`.

```python
from drst import get_strategy, build_all_curves
curves = build_all_curves(df, cfg, strategy=get_strategy("event_anchored"))
```

---

## Estensioni previste

In ordine di impatto atteso.

### 1. Fondo strutturale continuo dal tracking XY

La v1 ha fatto il primo passo: la difesa entra nello stress con `structural_vulnerability`. Ma quelle colonne sono campionate al `frame_start` della corsa, quindi restano un attributo **della corsa**, non un segnale di fondo continuo. `FrameIndex.baseline` è ancora zero.

Da aggiungere come `baseline`, calcolate frame per frame dal tracking XY già in locale:

- altezza della linea difensiva normalizzata per lato, `L_t` = media delle x dei 4 difensori più arretrati
- altezza relativa alla palla, `R_t = x_palla − L_t`
- compattezza longitudinale e laterale dei 10 di movimento
- numero di difensori fra palla e porta, velocità di recupero della linea

Riferimento: Teranishi et al., *Prediction-based evaluation of back-four defense with spatial control in soccer* ([arXiv:2511.06191](https://arxiv.org/pdf/2511.06191)).

L'enriched non ha compattezza né velocità di recupero della linea: quelle richiedono davvero i parquet.

Implementare: `indexing.DenseTrackingIndex`.

### 2. La curva separation(t) è buttata via

`separation_gain = end − start` sono due punti su una traiettoria di 25-35 frame. La pipeline `compute_separation(frames, run)` esiste già nella repo.

Feature che `gain` nasconde: separation minima e **quando** avviene, pendenza nel momento critico, area sotto una soglia, numero di inversioni.

Massimo guadagno per minimo sforzo: il codice di calcolo c'è.

Implementare: `stress.separation_curve_containment`.

### 3. Awareness (Filippo)

Interfaccia di scambio concordata:

```
frame | defender_id | attacker_id | awareness | in_view
```

Risolve l'ambiguità che la separation non scioglie: due corse con identico `separation_gain` sono diverse se in una il difensore guardava il runner (scelta tattica) e nell'altra no (errore). Attesa proprio dove il modello oggi cade — `cross_receiver` e `dropping_off`.

Disponibile su 2 match A-League con body pose; sui 18 match 2D serve una versione degradata (direzione di moto come proxy del cono visivo).

Implementare: `stress.awareness_containment`.

### 4. Prior geometrico non derivato dagli esiti

Fatto a metà nella v1: `spatial_priors` aggiunge la dimensione spaziale, ma resta un prior derivato dallo shot rate. Un prior da geometria pura (distanza dalla porta della zona di arrivo, xT della cella) eliminerebbe la circolarità residua e non richiederebbe il controllo per zona nella validazione.

Zona più fine di `third_start`: `channel_start` × `third_start` dà 15 celle invece di 3, ma va verificata la numerosità per cella.

### 5. Aggregazione non lineare

Lo stress di corse simultanee si somma. Ma il decoy è il 68% dei `lead_to_shot`: una corsa esca potrebbe *amplificare* l'altra, oppure essere ridondante. `aggregation="noisy_or"` è già implementata e testabile cambiando un parametro.

### 6. Opponent adjustment

Lo stress non sa chi attacca. Con 20 match e 13 squadre il mix di avversari è molto diverso fra squadre. Minimo: effetto casuale per squadra attaccante in un modello misto.

---

## Limiti da dichiarare nel paper

- **Copertura temporale.** Solo il 12% dei frame ha una corsa attiva; col decadimento si arriva al 40%. Oltre metà partita il segnale è zero *per costruzione*. Risolvere questo è l'estensione 1.
- **La v1 è per-corsa, non continua.** `structural_vulnerability` descrive la difesa al momento della corsa, non un fondo geometrico. `FrameIndex.baseline` resta zero.
- **Parte della stabilità v1 è composizionale.** Il prior spaziale fa dipendere lo stress da *dove* una squadra concede corse, che è a sua volta una proprietà stabile della squadra (un blocco basso concede corse nel proprio terzo). È una caratteristica difensiva legittima, ma non è la stessa cosa di "gestisce male i duelli".
- **Circolarità residua.** `danger_prior` deriva dallo shot rate per subtype e la validazione è su `lead_to_shot`. Il controllo per subtype la neutralizza (l'effetto misurato è *dentro* ogni subtype), ma un prior geometrico sarebbe più pulito.
- **Campione.** 8 squadre con ≥3 match. La stabilità +0.44 ha IQR [+0.13, +0.59]: è segnale, non certezza.
- **Attribuzione individuale.** `defender_profile` usa il nearest defender, che attribuisce a un solo giocatore una responsabilità spesso condivisa.
- **Frequenza del tracking.** `tracking_hz=10` è assunta, da verificare sui parquet SkillCorner A-League (la tesi lavorava a 25 Hz su Hawk-Eye).

## Test

```bash
pytest tests/ -q
```

27 test su kernel, containment, prior, curva, indicizzazione, aggregazione, prior spaziale e vulnerabilità strutturale. Fra questi `test_stress_ignores_outcomes` e `test_structural_ignores_outcomes` verificano che lo stress non cambi quando si invertono gli esiti — è l'invariante che rende la validazione non circolare.
