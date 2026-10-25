# Changelog

Ogni voce: cosa cambia, perché, e il numero che lo giustifica (o che lo smentisce).

## [0.2.0] — 2026-10-21

### F1 · Canali della risposta
- `exposure.py` divide già X in `X_opt` (opzioni di passaggio scoperte) e `X_car` (spazio al
  portatore). `expected.add_expected` stima ora un'attesa per canale (stesso contesto solo
  d'attacco, stessa GroupKFold) → `R_opt`, `R_car`. `rank.py` aggiunge `DRF_opt`, `DRF_car`.
- `validate.channel_test`: logistica con entrambi i canali, misurati nei 2 s prima dell'evento.
- Risultato (20 match): i canali sono ortogonali (r = 0.01) e fanno lavori diversi.
  | attacco | esito | OR opzioni | OR portatore |
  |---|---|---:|---:|
  | corsa | targeted / received | 0.99 / 0.98 (ns) | 1.03 / 1.02 (ns) |
  | corsa | tiro | **0.90** (p 0.002) | 0.91 (p 0.07) |
  | possesso | tiro | **0.94** (p 0.03) | 0.95 (ns) |
  | possesso | perdita palla | 1.01 (ns) | **1.24** (p 6e-9) |
  Coprire le opzioni evita i tiri; pressare il portatore fa recuperare palla.
- Split-half per squadra: `DRF_opt` **0.73**, `DRF` 0.61, `DRF_car` 0.53, tiri concessi 0.36.
  La copertura delle opzioni e' il tratto di squadra piu' stabile misurato finora.

### F3 · Delta per partita (leave-one-match-out)
- `validate.lomo_delta` / `lomo_all`: per ogni partita esclusa, logistica base e con i termini
  di risposta stimate sulle altre 19, valutate sulla partita esclusa.
  | test | log-loss migliora | AUROC migliora |
  |---|---:|---:|
  | corsa → tiro, + R (prima della corsa) | **14/20** | 13/20 |
  | corsa → tiro, + canali | 13/20 | **15/20** |
  | corsa → ricevuta, + R | 1/20 | 7/20 |
  | possesso → tiro, + R | 9/20 | 10/20 |
  | possesso → perdita palla, + canale portatore | 13/20 | **16/20** (ΔAUROC mediano +0.038) |
- Lettura: la dissociazione regge anche partita per partita (ricevuta 1/20). Sui possessi la
  risposta **non** generalizza al tiro: il risultato pooled (OR 0.88) era trainato da poche
  partite. Va detto nel paper.

### F4 · Carico contro risposta
- `load.py`: carico per squadra-partita dallo stress di `drst` (`runs_with_stress.parquet`),
  con fallback interno su X̂ medio.
- `figures.fig_load_response` (`docs/fig3_load_vs_response.png`): 13 squadre su due assi,
  piu' il pannello "con cosa correla la metrica di oggi".
- Squadra-partita, Spearman con il carico di drst: **tiri concessi +0.71**, DRF −0.32.
  I tiri concessi misurano soprattutto quanto sei stato attaccato.
- Etichette dirette con piazzamento anti-collisione (`_place_labels`).

### F6 · Migliore risposta osservata (`analogs.py`)
- Per ogni attacco: analoghi = stesso tipo e subtype, contesto d'attacco simile all'inizio
  (feature di X̂ + posizione del protagonista), X̂ entro un fattore 1.5, **altra partita e
  altra squadra**. Migliore = R piu' alta (mai scelto per esito). Meno di 5 analoghi =
  "nessun riferimento".
- Supporto con il vincolo di subtype (prima stima di ottobre era senza): corse 67%
  (critiche **61%**), possessi 64% (critici solo **21%**).
- `validate.analog_test`: il confronto con gli analoghi spiega i tiri **meglio** dell'attesa
  del modello. AIC corse 3531 (R) → 3516 (analoghi); possessi 2905 → 2890.
  Insieme nello stesso modello l'effetto passa agli analoghi: OR 0.74 (p 0.002) corse,
  0.61 (p < 1e-5) possessi. R e confronto con gli analoghi correlano 0.78–0.81.
- Lettura: il riferimento osservato e' un'attesa locale piu' fine (stesso subtype, stessa
  posizione del protagonista). E' anche quello che rende il video prescrittivo.

### F2 · Placebo: la difesa di un altro momento (`placebo.py`)
- Per ogni frame le posizioni della squadra che difende sono sostituite con le sue posizioni
  di 90 s (e 300 s) prima, nello stesso tempo; attacco, palla e possesso restano veri.
  Ricalcolo completo: geometria, esposizione, attesa OOF, risposta, test.
- Tutti gli effetti tornano a ~1:
  | test | reale | placebo 90 s | placebo 300 s |
  |---|---:|---:|---:|
  | corsa → tiro, R prima della corsa | 0.86 (p 0.001) | 1.00 (p 0.99) | 1.01 (p 0.91) |
  | possesso → tiro, R prima | 0.93 (p 0.02) | 1.01 (p 0.77) | 1.05 (p 0.15) |
  | corsa → tiro, canale opzioni | 0.90 (p 0.002) | 1.01 (p 0.91) | 1.01 (p 0.93) |
  | possesso → perdita, canale portatore | 1.24 (p 6e-9) | 1.01 (p 0.79) | 1.03 (p 0.05) |
- R non sta leggendo l'attacco ne' la forma media della squadra: serve la difesa di quel
  momento. `scripts/run_placebo.py`.

### F7 · Video per lo staff (`staff_video.py`, `cli video --style staff --lang it|en`)
- Una sola misura in parole: **spazio concesso come multiplo del normale** (X / X̂),
  e i due canali con lo stesso linguaggio (spazio al portatore, linee di passaggio scoperte).
  Fasce: < 0.8× meglio, 0.8–1.25× nella norma, > 1.25× peggio; saturazione mostrata come ≥4×.
- In campo solo cio' che serve: protagonista con nome e scia, il suo marcatore, le due opzioni
  piu' scoperte; le altre marcature sono linee sottili. Legenda colori-squadre in alto.
- Rallentatore sull'azione e **fermo immagine** nel momento di massimo spazio concesso con
  al massimo tre frasi automatiche ("#44 nessuno lo marca · il piu' vicino #17 a 12 m").
- Scheda iniziale con il verdetto, poi la clip, poi **come l'ha difeso la squadra con la
  miglior risposta osservata** e una scheda di confronto affiancata con il motivo della
  differenza. Senza analoghi sufficienti lo dice ("nessun riferimento affidabile").
- Per ogni episodio anche il fermo immagine PNG per le slide e un indice CSV con la lettura.
- Gli episodi scelti privilegiano quelli critici con un riferimento disponibile.

### Packaging
- `reproduce.sh` (dati → test → pipeline → placebo → figure → video), `LICENSE` MIT, v0.2.0.

### F5 · Test su scene costruite a mano (`tests/test_scenes.py`)
- 2 contro 1, difensore sulla linea di passaggio, pressione vicina/lontana, retropassaggio a
  valore zero, lato porta, isteresi della marcatura, canali che sommano al totale.
- **Trovato un limite reale**: un runner alle spalle di una linea piatta risulta coperto,
  perche' il modello conosce solo il passaggio rasoterra (un passaggio a terra attraverso
  un varco di 10 m e' davvero intercettabile). Tenuto come `xfail(strict=True)`: quando
  arrivera' il passaggio alto il test passera' e segnalera' di togliere lo xfail.

## [0.1.0] — 2026-10-20 · baseline

- Geometria frame per frame: time-to-intercept, copertura delle opzioni (punto di ricezione
  o linea di passaggio), pressione sul portatore, marcatura ungherese lato porta.
- Esposizione X(t), attesa X̂(t) dal solo attacco (GroupKFold su match), risposta R = X̂ − X.
- Attacchi subiti (`player_possession`, `off_ball_run`) con pericolo in z-score.
- Ranking interno alle partite: episodi, squadre (DRF), difensori (LOO, aggiustato per ruolo).
- Video 2D degli episodi critici.
- Risultati: OR risposta su tiro 0.83–0.88 (0.86 pre-evento); split-half DRF 0.61;
  nullo frame-level ΔAUROC +0.0005 [−0.0007, +0.0017].
