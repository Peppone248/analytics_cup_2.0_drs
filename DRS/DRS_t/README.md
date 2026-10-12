# drst — DRS(t)

> Stato: 12 ottobre 2026 · 8877 corse su 20 match A-League 24/25 · tracking a copertura 100%
> Numeri di riferimento riprodotti con la config di default: vedi [Risultati di riferimento](#risultati-di-riferimento).

Defensive Response Signature come **curva temporale** invece che come residuo di un modello sugli esiti.

Ogni off-ball run è uno **stress test** che l'attacco applica alla difesa. Lo stress si accumula e decade: `DRS(t)` è lo stress difensivo istantaneo.

```
stress = danger_prior × containment_failure × structural_vulnerability
```

- `danger_prior` — `exp(−dist/25 m)` dalla porta attaccata, al punto di **partenza** della corsa. Geometria pura: non tocca gli esiti.
- `containment_failure` — `separation_gain` z-scorato per subtype, poi sigmoide.
- `structural_vulnerability` — media geometrica pesata di `delta_to_last_defensive_line`, `n_simultaneous_runs`, `penalty_area_start`, più `str_stretch_rate` e `str_recovery_rate` dal tracking XY.

Lo stress **non usa mai gli esiti** (`targeted`, `received`, `lead_to_shot`): restano fuori per la validazione.

Due test coprono due fallimenti diversi e servono **entrambi** — vedi [Due test, due fallimenti](#due-test-due-fallimenti):

| test | cosa esclude |
|---|---|
| dissociazione su `targeted`/`received` | che il DRS misuri la *scelta dell'attaccante* invece dello stato difensivo |
| split-half per squadra | che il DRS misuri l'*episodio* invece di un tratto riproducibile della squadra |

---

## Uso — file unico

Tutto sta in `drst.py`. Nessuna installazione: basta `pip install -r requirements.txt`
(`pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `pyarrow`; `matplotlib`
solo per le figure).

I parametri del run stanno nel **blocco di configurazione in testa a `drst.py`**:
percorsi di `layer3_enriched.csv` e `tracking_features.csv`, cartella di output,
`GEO_USE_ENDPOINT`, `USE_TRACKING_STRUCTURE`. Si modifica lì e si lancia il file,
da VS Code (`.vscode/launch.json` è già configurato) o da terminale:

```bash
python drst.py
```

Il prior di default è **geometrico**, quindi non dipende dalle colonne di zona.
Quello che l'input decide è il **terzo fattore**:

| input | colonne | stress |
|---|---|---|
| `layer3_all_matches.csv` | 18 | `prior × containment` |
| `layer3_enriched.csv` | 47 | `prior × containment × struttura (enriched)` |
| `+ tracking_features.csv` | +19 | `prior × containment × struttura (enriched + tracking)` |

I test del file unico:

```bash
pytest test_drst.py -q
```

Questo file è **generato** da `tools/build_single_file.py` nel package: non
modificarlo a mano, le modifiche vanno nel package e poi si rigenera.

## Il terzo fattore: vulnerabilità strutturale

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

**8877 run, 20 match A-League 24/25, `layer3_enriched.csv` + `tracking_features.csv` a copertura 100%, config di default** (prior geometrico sulla partenza, struttura con tracking, curva sep(t) off).

Riprodurre con:

```bash
python -m drst.cli --layer3 data/layer3_enriched.csv \
  --tracking-features data/tracking_features.csv --out out/ref --oos
```

### Dissociazione — il risultato principale

`DRS(t)` misurato nei 3 secondi **prima** di una corsa, logistica che controlla per subtype. Out-of-sample = GroupKFold su `match_id`.

| esito | OR (OOS) | IC 95% | p |
|---|---:|---|---:|
| `targeted` | 0.993 | [0.949, 1.039] | 0.76 |
| `received` | 1.019 | [0.971, 1.070] | 0.44 |
| **`lead_to_shot`** | **1.217** | **[1.155, 1.282]** | **2.1e-13** |

In-sample e OOS coincidono (1.2167 vs 1.2171): conseguenza diretta del fatto che il prior geometrico non dipende dal training.

Il `DRS(t)` non predice se il portatore mira la corsa né se la riceve — quelle sono **scelte dell'attaccante**. Predice fortemente se quella corsa genera un tiro.

È il test che dimostra che il `DRS(t)` misura lo **stato difensivo** e non la scelta dell'avversario: il difetto che aveva affondato il modello di stadio 1 `p(targeted | run)`.

**Nota sul confronto con le versioni precedenti.** Gli OR di questo README passati per 1.257 (prior per subtype), 1.175 (prior per subtype × zona) e 1.235 (prior geometrico senza struttura da tracking) **non sono confrontabili fra loro**: ogni prior cambia anche i controlli obbligatori della logistica. Con il prior per zona serviva `dissociation(..., extra_controls=["third_start"])`, altrimenti parte dell'effetto misurato era il prior stesso; con il prior geometrico quel controllo non serve più. Il numero di riferimento è **1.217**, l'unico prodotto dalla config di default sui dati attuali.

### Gradiente grezzo

| DRS(t) pre-run | n | shot rate | targeted | received |
|---|---:|---:|---:|---:|
| zero | 2154 | 7.4% | 36.1% | 28.8% |
| Q1 | 1681 | 11.3% | 33.4% | 27.3% |
| Q2 | 1681 | 11.1% | 33.4% | 25.8% |
| Q3 | 1680 | 10.6% | 34.4% | 26.1% |
| Q4 | 1681 | **21.3%** | 33.7% | 25.3% |

Shot rate ×2.87 da zero a Q4, `targeted` e `received` piatti come sempre.

**Il gradiente non è monotono.** Con il prior per subtype × zona lo era (9.8 → 11.3 → 12.9 → 20.4). Con il prior geometrico Q1–Q3 sono piatti intorno all'11% e tutto il segnale è nel **salto di Q4**. Non contraddice l'OR della logistica — che è stimato su `drs_z` continuo, non sui quartili — ma cambia la lettura: il DRS(t) non è una scala graduata di pericolo, distingue bene **lo stato di stress alto da tutto il resto**. Per il paper è una figura da presentare come soglia, non come gradiente.

Il controllo per subtype tiene su `behind` (1.50×), `support` (1.43×), `run_ahead_of_the_ball` (1.37×), `coming_short` (1.80×). **Cade** su `cross_receiver` (0.99×, pericolo già saturo) e `dropping_off` (0.57×, subtype semanticamente eterogeneo). Sono i due casi in cui la separation da sola non discrimina e serve l'awareness.

### Stabilità per squadra

400 split casuali (`seed=0`, quindi **deterministico**: due run della stessa config danno lo stesso numero), 8 squadre con ≥3 match.

| metrica | mediana | IQR | % positivi |
|---|---:|---|---:|
| `n_runs` (volume) | +0.695 | [+0.578, +0.804] | 99% |
| **`stress_per_run`** | **+0.724** | **[+0.600, +0.821]** | **100%** |
| `drs_mean_all` | +0.713 | [+0.587, +0.808] | 99% |
| `drs_mean_supported` | +0.682 | [+0.525, +0.785] | 97% |
| `containment_failure_mean` | +0.026 | [−0.281, +0.190] | 52% |

Per confronto, i residui del modello di stadio 1 `p(targeted|run)`: **−0.128**. Storicamente: +0.438 con il prior per subtype, +0.627 con il prior per subtype × zona, +0.724 oggi col geometrico.

`stress_per_run` al 100% di split positivi è il risultato più solido del modello: il tratto di squadra si riproduce in ogni partizione provata.

**Attenzione al volume.** `drs_mean_all` correla 0.83 con `n_runs`: residualizzando sul volume crolla a −0.15. Quella stabilità è quasi tutta "quanto ti attaccano", non "quanto difendi bene". Per i profili di squadra usare solo `stress_per_run`, che correla 0.489 con `n_runs` e residualizzato resta intorno a +0.42.

`containment_failure_mean` resta fermo a **+0.026 con il 52% di split positivi**: la componente puramente difensiva, isolata dal prior di pericolosità, è dentro il rumore. Non è un dettaglio da sistemare, è **il limite aperto del modello** — vedi [Il tetto del modello](#il-tetto-del-modello).

### Varianza dello stress

Contributo di ciascun fattore alla varianza di `log(stress)`:

| fattore | quota |
|---|---:|
| `danger_prior` | 51% |
| `structural_vulnerability` | 23% |
| `containment_failure` | 26% |

Metà della varianza è geometria del punto di partenza. È coerente con il quadro generale: quello che il modello sa fare bene è dire *dove e quando* la difesa è esposta, non *quanto bene il singolo duello è stato gestito*.

### Da verificare: la struttura dal tracking paga?

Risultato nuovo, non ancora deciso. Ora che `str_stretch_rate` e `str_recovery_rate` sono a copertura 100% su tutti i 20 match (prima erano validate su **un solo match**, 387 corse), includerle **peggiora leggermente entrambe le metriche**:

| | OR (OOS) | split-half `stress_per_run` | varianza struttura |
|---|---:|---:|---:|
| `use_tracking_structure=True` *(default)* | 1.217 | +0.724 | 23% |
| `use_tracking_structure=False` | **1.234** | **+0.749** | 33% |

Il costo è −0.017 di OR e −0.025 di split-half. Piccolo ma coerente sulle due metriche, e nella direzione sbagliata rispetto alla correlazione parziale positiva (+0.071 e +0.063) misurata sul singolo match. `drs_mean_all` va nell'altro verso (+0.713 con, +0.705 senza), ma `stress_per_run` è la metrica che decide.

L'ipotesi più probabile è la stessa del punto di arrivo della corsa: `stretch_rate` e `recovery_rate` descrivono **come è andato quell'episodio**, non come la squadra si dispone. Da verificare prima di toccare il default.

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
├── plotting.py          figure statiche
├── animate.py           animazione 2D su campo
├── tracking_features.py curva separation(t) e struttura difensiva dal XY
├── evaluate_upgrade.py  confronto fra configurazioni del modello
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

## Animazione 2D

```bash
python -m drst.cli --layer3 data/layer3_enriched.csv --out out/v1 \
  --animate-match 1996436 \
  --tracking E:/Projects/opendata/data/matches/1996436/1996436_tracking_extrapolated.jsonl \
  --n-windows 5 --spatial-mode gap
```

Produce in `out/v1/clips/` un MP4 per finestra (GIF se ffmpeg manca): campo con heatmap di stress, giocatori, palla, linee tratteggiate difensore-runner sulle corse attive, e sotto la curva DRS(t) con cursore sul frame.

### I tre modi spaziali

Il DRS(t) è uno scalare, il campo è bidimensionale: serve decidere *dove* sta lo stress di ogni corsa. Il campo somma esattamente al DRS(t) scalare in tutti e tre i casi — la heatmap ridistribuisce lo stress nello spazio, non lo inventa (`test_active_weights_matches_kernel` lo verifica frame per frame).

**`runner`** — *dov'è la minaccia adesso*

Gaussiana (σ 5 m) centrata sulla posizione del runner. È una mappa di presenza pesata: risponde a "quali giocatori sono pericolosi in questo istante", senza dire dove stanno andando né chi li marca. Usalo per verificare che la pipeline funzioni — il calore deve stare sopra i pallini.

**`projected`** — *dove sarà la minaccia fra poco*

Stessa gaussiana, spostata lungo il vettore velocità del runner:

```
posizione_futura = posizione + velocità × project_s        (default 1.5 s)
```

La velocità è la differenza di posizione fra il frame corrente e quello di un secondo prima. Un giocatore fermo resta dov'è; uno che scatta a 7 m/s si sposta di ~10 m, e il calore entra in area mentre i corpi sono ancora fuori. È il modo che anticipa.

*Limite*: la velocità su un secondo è rumorosa, e un attaccante che cambia direzione viene proiettato dove non andrà. Funziona su corse lineari (`behind`, `run_ahead_of_the_ball`), meno su corse curve (`pulling_wide`, `overlap`).

**`gap`** *(default)* — *dove si sta aprendo il varco*

Lo stress è depositato in cinque punti lungo il **segmento difensore-runner**, con intensità crescente verso il runner (dal 30% al 100%). Il presupposto tattico: il pericolo non è dove sta l'attaccante, è nello spazio fra lui e chi dovrebbe marcarlo. Difensore addosso → segmento corto, calore concentrato. Difensore a quindici metri → il calore si allunga e disegna il corridoio aperto.

| | usalo per |
|---|---|
| `runner` | debug della pipeline |
| `projected` | hero case su corse in profondità, quando il punto narrativo è l'anticipo |
| `gap` | paper e lavoro analitico: è la lettura che un allenatore riconosce |

I tre non si escludono. La stessa sequenza renderizzata nei tre modi è una figura interessante di per sé: mostra che il modello ha una lettura spaziale, non solo un numero.

### Come si legge un clip

- **Colorbar** a destra del campo: 0 → `vmax`. Il nero è annotato come *non misurato*, perché con l'indicizzazione event-anchored il 63% dei frame è a zero per assenza di eventi, non perché la difesa sia in controllo.
- **Contorno giallo** attorno a ogni corsa attiva: perimetra il contributo della **singola** corsa prima della somma. Senza, due corse vicine si fondono in una macchia unica e non si distinguono — ed è proprio il caso dell'overload, il segnale più forte del modello.
- **Pastiglia numerata** sul runner, con subtype e peso. Gli stessi numeri del pannello laterale.
- **Pannello CORSE ATTIVE**: fino a 5 corse ordinate per peso, ciascuna con la scomposizione nei tre fattori — `pericolo` (prior del subtype e zona), `duello` (quanto il difensore ha perso terreno rispetto al normale per quel subtype), `struttura` (posizione rispetto alla linea, corse simultanee, area). Risponde a *perché* questa corsa pesa più di quella: due corse con peso simile possono averlo per ragioni opposte.
- **Linea tratteggiata** difensore-runner, opacità proporzionale al peso.
- **Curva in basso** con cursore sul frame: dà il contesto temporale che il campo da solo non ha.

Il flusso di lettura: prima la curva (stai salendo, sei al picco, stai scendendo?), poi il campo (*dove*), poi il pannello (*perché*), e se il caso è interessante `runs_with_stress.parquet` per i numeri esatti.

**Attenzione alla scala**: `vmax` è calcolato una volta su tutte le finestre del match e riusato, quindi due clip dello stesso match sono confrontabili. Due clip di **match diversi** no — `vmax` è ricalcolato per match. Per confrontare squadre serve fissare un `vmax` globale a mano.

**Il team_id non è nel tracking SkillCorner.** Lo ricaviamo dal Layer 3 (ogni runner appartiene a `team_id`, ogni difensore a `defending_team_id`). Restano fuori i due portieri, assegnati per prossimità ai compagni già noti. Se hai il `match.json` dell'opendata, la mappa esatta `players[i].id -> team_id` è preferibile.

**Avvertenza di lettura**: con l'indicizzazione event-anchored il 63% dei frame è a zero perché *non misurato*, non perché la difesa sia in controllo. `find_peak_windows` seleziona per costruzione dove c'è supporto, ma una finestra scelta a mano può non averlo.

## Feature dal tracking XY

Due gruppi, estratti una volta sola in un CSV che si fa merge sul Layer 3.

```bash
python -m drst.tracking_features \
  --layer3 data/layer3_enriched.csv \
  --tracking-root E:/Projects/opendata/data/matches \
  --out data/tracking_features.csv

python -m drst.cli --layer3 data/layer3_enriched.csv \
  --tracking-features data/tracking_features.csv --out out/v2
```

Costo: ~4 secondi per match, zero curve mancanti sul match testato.

### Curva separation(t)

`separation_gain = end − start` sono due punti su 25-35 frame. La curva intera dà `sep_min`, `sep_min_frac` (*quando* avviene il minimo), `sep_slope_min`, `sep_time_close`, `sep_inversions`, `sep_auc`, `sep_late`, `sep_range`.

`separation_curve_containment` le usa combinando il gain classico con la **fuga tardiva** `sep_late − sep_min`: quanto il runner si libera *dopo* il momento di massima prossimità, che è quando arriva la palla.

### Struttura difensiva

Compattezza longitudinale e laterale dei giocatori di movimento, altezza e slabbratura della linea a quattro, e la loro evoluzione: `str_stretch_rate` (quanto la difesa si allunga durante la corsa) e `str_recovery_rate` (quanto si ricompone nei 2s dopo). Quest'ultima non è ottenibile dall'enriched in nessun modo.

**Orientamento**: il tracking è in coordinate assolute e le squadre cambiano campo all'intervallo, mentre `last_defensive_line_x_start` dell'enriched è già normalizzato. Senza orientare per `attacking_side`, la linea calcolata qui è scorrelata da quella di SkillCorner (r = −0.17). Orientata: **r = 0.994**.

### Cosa si è verificato, e cosa no

Su un **singolo match** (1996436, 387 corse — campione piccolo, da riverificare sui 20):

| | corr. grezza | corr. parziale (netto di fase e subtype) |
|---|---:|---:|
| `str_spread_x` | −0.224 | **+0.010** |
| `str_line_spread` | −0.302 | −0.089 |
| `str_area` | −0.258 | −0.086 |
| `str_stretch_rate` | +0.087 | **+0.071** |
| `str_recovery_rate` | +0.029 | **+0.063** |

La compattezza assoluta è quasi interamente un proxy della fase: `str_spread_x` crolla a zero controllando. Le due di evoluzione (`stretch_rate`, `recovery_rate`) sono le uniche la cui correlazione *sale*: informazione che la fase non contiene. Solo quelle due entrano in `structural_vulnerability`.

Sulla curva separation(t) il verdetto è ambiguo:

- come **regressori liberi** in una logistica non aggiunge nulla: AIC 301.0 contro 297.6 del baseline, sei parametri per niente (la struttura invece scende a 291.4)
- dentro lo **stress**, come trasformazione vincolata e normalizzata per subtype, aiuta: corr. stress↔esito da +0.230 a +0.247, e con la struttura a +0.290

**Verificato sui 20 match: la curva peggiora il modello.** `separation_curve_containment` è quindi **disattivato di default** (`cfg.use_curve_containment = False`), resta attivabile per sperimentare.

| | `containment_failure_mean` | `stress_per_run` | OR |
|---|---:|---:|---:|
| A riferimento | +0.026 | 0.628 | 1.165 |
| B + struttura tracking | +0.026 | **0.637** | 1.148 |
| C + curva sep(t) | **−0.134** | 0.607 | 1.165 |
| D completo | **−0.134** | 0.622 | 1.148 |

Un containment a −0.134 con il 32% di split positivi è anticorrelazione: le squadre con containment alto in una metà dei match lo hanno basso nell'altra, cioè regressione alla media. Più risoluzione temporale sul duello non aiuta.

La lettura: il problema non è che la separation sia misurata male con due punti invece di trenta. È che **la separation non distingue una scelta tattica da un errore** — il difensore che lascia andare il runner perché lo vede e ha copertura produce la stessa curva di quello che lo perde.

Confermato da un terzo tentativo: feature di *accoppiamento reattivo* fra le traiettorie di difensore e runner (correlazione a lag delle velocità, angolo di ingaggio, ritardo di partenza) hanno correlazioni parziali sotto 0.09 e peggiorano l'AIC (291.5 → 298.0). E sullo stesso match `separation_gain` da solo non aggiunge nulla a fase + subtype (291.5 → 293.3).

**Il containment serve comunque.** Togliendolo del tutto dallo stress, la stabilità scende (misurato senza la struttura dal tracking, quindi la riga "con containment" è il +0.749 di quella config, non il +0.724 del default):

| | OR | split-half | residualizzato |
|---|---:|---:|---:|
| con containment | 1.235 | **0.749** | **0.409** |
| senza (costante 0.5) | 1.252 | 0.666 | 0.319 |

Quindi non predice l'esito ma porta informazione di squadra riproducibile. Spiega il paradosso di `containment_failure_mean` fermo a +0.026: la *media* del containment per squadra non è stabile, il containment **pesato per la pericolosità** sì. Pesa il 26% della varianza dello stress (prior 51%, struttura 23%).

### Rivalutare il modello

```bash
python -m drst.evaluate_upgrade \
  --layer3 data/layer3_enriched.csv \
  --tracking-features data/tracking_features.csv \
  --out out/upgrade --oos
```

Confronta quattro configurazioni (v1, + struttura, + curva, completo) sulle stesse corse. Cosa guardare, in ordine:

1. **split-half di `stress_per_run`** — riferimento della config di default **+0.724**, residualizzato sul volume ≈ +0.42. È il test che decide.
2. **`containment_failure_mean`** — fermo a +0.026: è quello che la curva dovrebbe sbloccare. Se resta fermo anche in C e D, il problema non è la risoluzione della separation (ed è esattamente cosa è successo).
3. **OR su `lead_to_shot`** — riferimento **1.217**, in-sample e OOS. Gradito ma secondario.
4. **La dissociazione deve reggere**: se `targeted` o `received` diventano significativi, quella configurazione sta reintroducendo la scelta dell'attaccante e va scartata a prescindere dal resto.

## Sweep di miglioramento (11 ottobre, 8877 corse)

Quattro direzioni testate sistematicamente. Una adottata, tre scartate con i numeri.

### Adottato: prior geometrico

`danger = exp(−dist / 25 m)` dove `dist` è la distanza del punto di partenza della corsa dalla porta attaccata. **Non tocca gli esiti**, quindi la circolarità sparisce e la validazione non ha più bisogno di controlli di zona.

| prior | OR (OOS) | split-half `stress_per_run` |
|---|---:|---:|
| subtype | 1.248 | 0.518 |
| subtype × third | 1.165 | 0.628 |
| **geometrico 25 m** | **1.234** | **0.749** |

Questo confronto è stato fatto **senza** la struttura dal tracking, che allora copriva un solo match. Con la config di default attuale il prior geometrico dà 1.217 e +0.724: il confronto fra prior non cambia verso, ma i numeri di riferimento del modello sono quelli in [Risultati di riferimento](#risultati-di-riferimento).

In-sample e OOS coincidono quasi (1.2353 vs 1.2343): conseguenza diretta del fatto che il prior non dipende dal training. La dissociazione regge meglio — `p(received)` passa da 0.35 a 0.68.

Il prezzo: il prior geometrico ignora il tipo di corsa, un `dropping_off` e un `behind` che partono dallo stesso punto hanno lo stesso valore. La variante ibrida (geometrico × prior del subtype) sta in mezzo senza nessuno dei due vantaggi pieni: OR 1.216, split-half 0.671. `prior_kind` in `DRSConfig` permette di tornare agli altri due.

### Scartato: opponent adjustment

Residualizzare `stress_per_run` sulle dummy della squadra attaccante porta la stabilità da +0.434 a +0.566. Ma sono **40 righe squadra-match su 12 dummy**, e un test di permutazione mostra che non è distinguibile da un artefatto: con etichette casuali la mediana è +0.220 e la distribuzione nulla è [−0.13, +0.65], con il valore vero all'88° percentile.

Non adottato. Serve molto più campione per validarlo — è una delle cose che un dataset più ampio sbloccherebbe.

### Scartato: zone più fini

`channel_start × third_start` dà 15 zone invece di 3, 84 celle con mediana 80 osservazioni. L'OR sale appena (1.175 → 1.181) ma la **stabilità scende** (0.644 → 0.625, residualizzata 0.420 → 0.399): più celle, prior più rumorosi. Irrilevante comunque, visto che il prior geometrico ha sostituito quello per zona.

### Scartato: aggregazione non lineare

| | OR |
|---|---:|
| `sum` (default) | 1.175 |
| `max` | 1.155 |
| `noisy_or` | 1.181 |

`max` peggiora, `noisy_or` migliora di 0.006 — dentro il rumore. L'ipotesi che l'overload non fosse additivo non si conferma. Restano selezionabili da `cfg.aggregation`.

## Prior geometrico: punto di arrivo della corsa

`danger_prior = exp(-dist/25)`, con `dist` la distanza dalla porta attaccata.
Fino alla v2 era la distanza del **punto di partenza**, l'unico che il Layer 3
enriched contiene. Ma una corsa in media avvicina alla porta di **6.1 m**:
il prior sulla partenza sottostima chi attacca la profondita'.

`tracking_features.run_geometry` estrae ora anche il punto di arrivo e la
palla, con lo stesso riallineamento `attacking_side` validato per la
struttura (sanity check: `corr(geo_x_start, x_start) = 1.000`). Il prior puo'
usare `geo_dist_min = min(partenza, arrivo)` con `geo_use_endpoint=True`.

Per la singola corsa e' chiaramente meglio. Sul match 1996436 (387 corse),
dove e' stato esplorato:

| punto usato | corr con shot | Q1 -> Q4 |
|---|---:|---|
| partenza | +0.403 | 3.1% -> 38.1% |
| arrivo | +0.458 | 0.0% -> 41.2% |
| media | +0.445 | 0.0% -> 39.2% |
| **min(partenza, arrivo)** | **+0.469** | 0.0% -> 41.2% |

Sui 20 match (8877 corse) le correlazioni sono molto piu' basse in assoluto
ma il verso e' lo stesso: `corr(danger_prior, lead_to_shot)` passa da
**+0.228** a **+0.342**, e `corr(stress, lead_to_shot)` da +0.184 a +0.264.
Quelle da citare nel paper sono queste, non le del singolo match.

### Perche' il default e' comunque OFF

A copertura piena sui 20 match, con la config di default, il punto di arrivo
migliora la corsa e peggiora la squadra:

| metrica | partenza *(default)* | min(start,end) |
|---|---:|---:|
| OR su `lead_to_shot` (OOS) | 1.217 | **1.254** |
| split-half `stress_per_run` | **+0.724** | +0.628 |
| split-half `drs_mean_all` | **+0.713** | +0.598 |

(Con `use_tracking_structure=False` le due colonne diventano 1.235/+0.749 e
1.255/+0.641: la conclusione non cambia, ma i due parametri vanno confrontati
a config fissa — mescolarli e' il modo piu' facile di sbagliare il confronto.)

La dissociazione regge in entrambi i casi (`targeted` p=0.82, `received`
p=0.30), quindi il test di dissociazione **non vede** questo problema: e' un
test sulla scelta dell'attaccante, non sulla stabilita'.

Tre ipotesi scartate con i dati, in ordine:

1. **Endogeneita' rispetto al containment** — se il punto di arrivo fosse
   "dove la difesa ha fermato la corsa", il prior conterrebbe il secondo
   fattore. Smentita: `corr(geo_progress, containment) = -0.004`.
2. **Doppio conteggio col terzo fattore** — `corr(prior, struttura)` passa
   da +0.537 a +0.532. Nessuna sovrapposizione aggiunta.
3. **Riparametrizzazione della scala** — la correlazione fra i due prior
   satura a **0.843** per qualunque scala fra 25 e 45 m, e non sale. E'
   informazione nuova, non `geo_prior_scale` travestita.

Quello che resta e' la spiegazione giusta: **dove** una corsa finisce e' una
proprieta' dell'episodio — se l'attaccante ha insistito, se la corsa di un
compagno ha portato via un marcatore — mentre **dove parte** e' una
proprieta' di come la squadra si dispone, e quella si ripete partita dopo
partita. Il punto di arrivo porta segnale sulla corsa e rumore sulla squadra.

Quindi dipende dall'unita' di analisi, e il default segue la claim
principale del DRS, che e' per squadra:

- **metriche di squadra, split-half, tabella del paper** -> OFF
- **animazione, case study, qualunque lettura per-corsa** -> ON, ed e' il
  prior migliore

Il gate sulla palla (`geo_ball_gate`, default off) aggiunge +0.001 di
correlazione: `ball_dist_goal_start` correla -0.52 con la distanza della
corsa, quindi duplica il prior invece di informarlo.

### Alternative scartate

Prior basati sull'**angolo sotteso dalla porta** (stile xG), su 8877 corse:

| prior | corr | spread Q5/Q1 |
|---|---:|---:|
| `exp(-d/25)` *(scelto)* | **0.228** | **6.19** |
| `exp(-d/15)` | 0.213 | 6.19 |
| `1/(1+d)` | 0.213 | 6.19 |
| `sqrt(angolo)` | 0.196 | 4.72 |
| `angolo x exp(-d/40)` | 0.188 | 5.56 |
| `angolo` | 0.187 | 4.72 |

A 30-50 m, dove parte la maggior parte delle corse, l'angolo sotteso e'
piccolo per tutti e satura: non discrimina.

---

## Estensioni previste

In ordine di impatto atteso.

### 1. Fondo strutturale continuo dal tracking XY

Il terzo fattore ha fatto il primo passo: la difesa entra nello stress con `structural_vulnerability`. Ma quelle colonne sono campionate al `frame_start` della corsa, quindi restano un attributo **della corsa**, non un segnale di fondo continuo. `FrameIndex.baseline` è ancora zero.

È anche la strada che potrebbe spiegare il risultato aperto sulla struttura dal tracking: se `stretch_rate` e `recovery_rate` peggiorano la metrica di squadra perché descrivono l'episodio, la stessa informazione calcolata come **fondo continuo** — media sulla fase di gioco, non sul singolo `frame_start` — potrebbe invece essere un tratto stabile.

Da aggiungere come `baseline`, calcolate frame per frame dal tracking XY già in locale:

- altezza della linea difensiva normalizzata per lato, `L_t` = media delle x dei 4 difensori più arretrati
- altezza relativa alla palla, `R_t = x_palla − L_t`
- compattezza longitudinale e laterale dei 10 di movimento
- numero di difensori fra palla e porta, velocità di recupero della linea

Riferimento: Teranishi et al., *Prediction-based evaluation of back-four defense with spatial control in soccer* ([arXiv:2511.06191](https://arxiv.org/pdf/2511.06191)).

L'enriched non ha compattezza né velocità di recupero della linea: quelle richiedono davvero i parquet.

Implementare: `indexing.DenseTrackingIndex`.

### 2. Awareness (Filippo) — l'unica strada che sblocca il containment

Interfaccia di scambio concordata:

```
frame | defender_id | attacker_id | awareness | in_view
```

Risolve l'ambiguità che la separation non scioglie: due corse con identico `separation_gain` sono diverse se in una il difensore guardava il runner (scelta tattica) e nell'altra no (errore). Attesa proprio dove il modello oggi cade — `cross_receiver` e `dropping_off`.

**Il problema di validazione.** Il body pose c'è su **2 match dei 20**, quindi la split-half per squadra — la claim principale del DRS — è impossibile su quel sottoinsieme. L'alternativa proposta è una **split-half a livello di giocatore** su circa 800 corse: si divide a caso l'insieme delle corse di ciascun difensore e si correla il suo containment nelle due metà.

Perché possa funzionare serve un numero minimo di corse per difensore. **Domanda aperta per Filippo: quanti difensori hanno almeno 10–15 corse nei 2 match con body pose?** Sotto quella soglia la split-half per giocatore è rumore e serve ripiegare su una validazione puramente qualitativa (case study sulle sequenze selezionate dal DRS).

Sui 18 match 2D serve una versione degradata (direzione di moto come proxy del cono visivo), che va validata separatamente contro quella da body pose sui 2 match in comune.

Implementare: `stress.awareness_containment`.

### 3. Opponent adjustment

Lo stress non sa chi attacca. Con 20 match e 13 squadre il mix di avversari è molto diverso fra squadre. Minimo: effetto casuale per squadra attaccante in un modello misto. **Già tentato e scartato** con il permutation test (vedi sotto): va ripreso solo con più campione, non con questi 20 match.

---

## Il tetto del modello

Con XY e `separation` il modello è al suo tetto, e il sintomo è sempre lo stesso numero: `containment_failure_mean` a **+0.026** con il 52% di split positivi. Otto strade indipendenti convergono su quel punto.

La ragione di fondo: **la separation non distingue una scelta tattica da un errore**, a nessuna risoluzione temporale. Il difensore che lascia andare il runner perché lo vede e ha copertura produce la stessa curva di quello che lo perde. Nessuna trasformazione di XY recupera quell'informazione, perché l'informazione non è nelle coordinate — è nella **direzione dello sguardo**.

Da cui: l'awareness frame-level non è un miglioramento fra tanti, è l'unico ingrediente nuovo in grado di muovere il secondo fattore.

### Strade scartate — indice

Tutte con un numero. **Non riproporle senza un motivo nuovo.**

| strada | numero che la chiude |
|---|---|
| opponent adjustment | permutation test: 88° percentile della distribuzione nulla, non distinguibile da artefatto |
| zone più fini (`channel × third`) | split-half 0.644 → 0.625 |
| aggregazione `max` | OR 1.155 contro 1.175 |
| aggregazione `noisy_or` | +0.006 di OR, dentro il rumore |
| curva separation(t) dentro il containment | `containment_failure_mean` +0.026 → **−0.134** |
| feature di coupling reattivo difensore-runner | corr. parziali < 0.09, AIC 291.5 → 298.0 |
| prior angolari stile xG | corr 0.187 contro 0.228 del geometrico |
| prior sul punto di arrivo `min(partenza, arrivo)` | split-half 0.724 → 0.628 |

### Due test, due fallimenti

Lezione metodologica da mettere nel paper, emersa dal caso del punto di arrivo.

| | cosa esclude | cosa **non** vede |
|---|---|---|
| dissociazione su `targeted`/`received` | che il DRS misuri la scelta dell'attaccante | che il DRS misuri l'episodio invece della squadra |
| split-half per squadra | che il DRS misuri l'episodio | che il DRS misuri la scelta dell'attaccante |

Il prior sul punto di arrivo **supera la dissociazione in entrambe le configurazioni** (`targeted` p=0.82, `received` p=0.30) e la split-half lo boccia. Un modello validato con la sola dissociazione l'avrebbe adottato. Servono entrambi i test, sempre, e vanno riportati entrambi.

---

## Limiti da dichiarare nel paper

- **Il containment non è riproducibile.** `containment_failure_mean` a +0.026 con il 52% di split positivi. Il secondo dei tre fattori, quello puramente difensivo, non è un tratto di squadra misurabile con questi dati. È il limite da dichiarare per primo.
- **Copertura temporale.** Solo il 12% dei frame ha una corsa attiva; col decadimento si arriva al 40%. Oltre metà partita il segnale è zero *per costruzione*. Risolvere questo è l'estensione 1.
- **Lo stress è per-corsa, non continuo.** `structural_vulnerability` descrive la difesa al momento della corsa, non un fondo geometrico. `FrameIndex.baseline` resta zero.
- **Il gradiente non è monotono.** Q1–Q3 piatti, salto a Q4. Il DRS(t) separa lo stress alto dal resto, non è una scala graduata.
- **Parte della stabilità è composizionale.** Il prior geometrico fa dipendere lo stress da *dove* una squadra concede corse, che è a sua volta una proprietà stabile della squadra (un blocco basso concede corse nel proprio terzo). È una caratteristica difensiva legittima, ma non è la stessa cosa di "gestisce male i duelli". Metà della varianza dello stress è quel prior.
- **Campione.** 8 squadre con ≥3 match. `stress_per_run` a +0.724 ha IQR [+0.600, +0.821]: è segnale solido, ma su 8 squadre.
- **Attribuzione individuale.** `defender_profile` usa il nearest defender, che attribuisce a un solo giocatore una responsabilità spesso condivisa.
- **Frequenza del tracking.** `tracking_hz=10` è assunta, da verificare sui parquet SkillCorner A-League (la tesi lavorava a 25 Hz su Hawk-Eye).
- **La struttura dal tracking è in discussione.** `use_tracking_structure=True` costa −0.017 di OR e −0.025 di split-half rispetto a tenerla fuori. Da chiudere prima della submission.

Non più un limite: la **circolarità**. Il `danger_prior` di default è geometria pura e non tocca gli esiti, quindi la validazione su `lead_to_shot` non ha più bisogno di controlli di zona. Gli invarianti `test_*_ignores_outcomes` lo verificano a ogni run dei test.

## Test

```bash
pytest test_drst.py -q
```

57 test su kernel, containment, prior (per subtype, spaziale, geometrico, punto di arrivo), curva, indicizzazione, aggregazione, vulnerabilità strutturale, feature dal tracking e animazione.

Gli invarianti che bloccano la circolarità — lo stress non deve cambiare quando si invertono gli esiti:

- `test_stress_ignores_outcomes`
- `test_structural_ignores_outcomes`
- `test_curve_containment_ignores_outcomes`
- `test_geometric_prior_ignores_outcomes`
- `test_geometric_prior_endpoint_ignores_outcomes`

Più `test_geometric_prior_endpoint_off_by_default`, che inchioda il default a `False` perché il punto di arrivo peggiora la metrica di squadra: se qualcuno lo riaccende, il test cade.
