# AWARENESS — activity log (Filippo)

Registro delle attività sul modulo awareness / "Ghost" per la SkillCorner × PySport Analytics Cup.
Una voce per giornata di lavoro, **la più recente in alto**. Ogni voce dice cosa è stato fatto, le
decisioni prese, i risultati e cosa resta aperto; i punti aperti aggiornati stanno nella voce più recente.
Come installare e lanciare il modulo: vedi [README.md](README.md).

Modello per una nuova voce:

```markdown
## AAAA-MM-GG — titolo breve

**Fatto**
- …

**Decisioni**
- …

**Risultati**
- …

**Aperto / prossimi passi**
- [ ] …
```

---

## 2026-10-05 — Viewer definitivo, difensore selezionabile, branch

**Fatto**
- **Viewer 3D rifatto** (template `viewer/pov3d_template.html`, output `outputs/ghost_viewer.html`):
  corpi volumetrici, ombre, ghost olografici, scie, campo con texture. La prima versione a stecchini è
  stata **eliminata**: questa è la versione definitiva. Dettaglio sotto.
- **Difensore focale selezionabile** nel viewer (menu, clic sul corpo, tasti `[` `]`). L'export ora salva
  le credenze di tutti i difensori di ogni clip.
- Corretti due difetti grafici emersi negli screenshot: colori slavati (mancava la conversione
  sRGB → lineare nei materiali) e ghost rossi che sul prato diventavano gialli (fusione additiva →
  normale).
- Ordine dei clip uguale tra `04_make_demo.py` e `05_rebuild_viewer.py` (`outputs/cache/clips_order.json`).
- **Git:** `git fetch`; `master` era già allineato a `origin/master`. Creato in locale il branch
  **`AWARENESS_filippo`** da `master` (analogo a `DRS_Beppe`). Nessun commit e nessun push.
- **Media non tracciati:** GIF e PNG dei clip in `outputs/media/clips/`, screenshot del viewer in
  `outputs/media/screenshots/`; `outputs/media/` e `.DS_Store` aggiunti a `AWARENESS/.gitignore`.
- README aggiornato; queste note trasformate in log per data.
- **Primi commit sul branch `AWARENESS_filippo` = versione v0** del modulo (libreria, script, viewer,
  output leggeri, documentazione). Non pushati. GIF e screenshot esclusi (`outputs/media/`).

**Decisioni**
- Quello committato oggi è una **v0**: funziona end-to-end ma è da rivedere (metodo, soglie, peso dei
  file, pulizia del codice) prima di considerarlo stabile.
- Il viewer v2 è l'unico viewer; la v1 non si mantiene.
- GIF e screenshot **non si committano per ora** (restano in `outputs/media/`).
- Prima di committare: pulizia del modulo.

**Viewer 3D — com'è fatto**
- **Camere:** Broadcast (orbitabile), Top, Follow (terza persona dietro al difensore), **POV** (camera
  tra le orecchie del difensore, orientata con l'head angle smussato, vignettatura ai bordi).
  Tasti 1–4.
- **Radar head-up** in POV: mostra ciò che il difensore non vede (dietro di lui) con ghost e linee.
- **Timeline + grafico** (w, ghost error, A_cov tesi) con banda colorata dello stato di credenza;
  spazio = play/pausa, frecce = frame per frame.
- **Corpi volumetrici** stile manichino: arti come cilindri rastremati con giunture, busto e bacino
  orientati sulle spalle/anche reali, maglia/pantaloncini/calzettoni nei colori squadra, scarpini, una
  piccola "visiera" sulla testa che indica la direzione dello sguardo. Il difensore focale ha la maglia
  leggermente illuminata e il numero in oro.
- **Ombre** in tempo reale; la luce segue l'azione. **I ghost non proiettano ombra**: sono ricordi.
- **Posa tenuta**: quando un giocatore esce dall'inquadratura (niente joint), si usa la posa del frame
  più vicino entro ±2 s, traslata sulla posizione attuale; oltre, una posa standard ruotata nella
  direzione di corsa. In entrambi i casi il corpo è semi-trasparente, così si vede che non è dato reale.
  È solo grafica: non entra nel modello.
- **Ghost olografici**: la sagoma ha la **posa dell'attaccante all'ultimo avvistamento** del difensore,
  spostata sulla posizione creduta (cioè "come lo ricorda"). Shader con bordo luminoso e scanline,
  colore secondo lo stato di credenza (verde/ambra/rosso); in stato *aware* il ghost si dissolve perché
  coincide con il giocatore. Anello a terra sotto il ghost, ellisse 95% sfumata (solo corridore), linea
  d'errore tratteggiata colorata per metri d'errore. Fusione normale, non additiva (sul prato verde
  quella additiva faceva sembrare gialli i ghost rossi).
- **Scie** degli ultimi 1.6 s del corridore (bianca) e del suo ghost (rossa): si vede il ghost che
  resta indietro e poi "scatta".
- **Allarme**: anello rosso pulsante sotto il corridore quando il difensore è in false belief.
- Campo con texture (strisce, grana, linee disegnate), porte con rete, cono visivo con sfumatura
  angolare/radiale (≈ la gaussiana del FoV), toggle per ghost, scie e numeri.
- **Difensore selezionabile**: menu a tendina accanto al clip (★ = più vicino al corridore a inizio
  run, il default), clic sul corpo di un difensore in 3D, o tasti `[` `]`. Ghost, cono, radar, grafico
  e sottotitolo passano al difensore scelto; il frame corrente resta lo stesso. Per farlo l'export
  salva le credenze di **tutti** i difensori (`frames[].G/H` allineati a `defs`, `series_by`): da qui i
  16.4 MB. Se il peso diventa un problema: arrotondare di più, sottocampionare a 12.5 fps, o caricare
  i dati di un difensore solo quando lo si seleziona (servirebbe un JSON separato e un piccolo server).
- Rigenerare dopo aver modificato il template: `scripts/05_rebuild_viewer.py` (usa i JSON in cache).

**Stato**
- Branch locale `AWARENESS_filippo` con i commit della v0 (non pushati).
- Tracciati: codice (`ghost/`, `scripts/`, `tests/`, `viewer/`), README, queste note,
  `requirements.txt`, `.gitignore`, `outputs/ghost_viewer.html` (16.4 MB), i due
  `offball_runs_*.parquet` e la calibrazione (`calibration_motion_prior.csv` / `.png`).
- Non tracciati: `.venv/`, `data_local/`, `outputs/cache/`, `outputs/media/`.

**Aperto / prossimi passi**
- [ ] Push del branch e condivisione con Beppe.
- [ ] Decidere se tenere `ghost_viewer.html` (16.4 MB) nel repo o pubblicarlo a parte (GitHub Pages /
      release / artifact): ogni rigenerazione aggiunge ~16 MB alla storia git.
- [ ] Verificare il **regolamento** della Cup (non è nella repo: `docs/index.md` contiene solo
      `../README.md`): uso dei dati esterni Hawk-Eye anche solo per validazione/calibrazione, formato
      della submission, limiti di pagine/video.
- [ ] Segnalare a Beppe i due branch remoti `DRS_Beppe` / `DRS_beppe` (differiscono solo per la
      maiuscola: su macOS può dare problemi) → cancellarne uno.
- [ ] Con Beppe: definizione operativa di pericolosità e scelta dei clip tra le situazioni DRS più
      pericolose.
- [ ] Metriche squadra/ruolo: *blind-side exposure* per sottotipo di run × fase difensiva (join con le
      phases of play).
- [ ] Versione "senza pose" (head angle = prior palla/velocità) su tutte le 20 partite, validata sulle 2
      con pose: quanto si perde?
- [ ] Calibrare anche σ_obs; sensitivity delle soglie (1.5 m, χ² 95%) e di `head_ok`.
- [ ] Viewer: export video (serve `brew install ffmpeg`), eventualmente "ghost di tutti i difensori".
- [ ] Se il file del viewer pesa troppo: arrotondare di più, 12.5 fps, o dati per difensore caricati a
      richiesta.

**Proposte da discutere con Beppe**
- *Integrazione con la DRS:* la DRS seleziona le situazioni (top-N per pericolosità, o per sottotipo ×
  fase difensiva); il modulo awareness restituisce per situazione e difensore `seen_frac`, `ghost_err`,
  `false_belief_s`. Join: `event_id` degli eventi dinamici (o `frame_start` 10 fps × 2.5). La body pose
  c'è solo per 2 partite su 20 → awareness = **deep dive**, DRS = pilastro (come già deciso in ROADMAP).
- *Metriche da paper / da club:* **ball-watching index** (% del tempo in cui il difensore più vicino non
  vede un corridore pericoloso; sui cross 35%), **blind-side exposure** per squadra, **leaderboard**
  individuale di false belief a parità di situazioni.
- *Validazione:* false belief prima della ricezione → `beaten_by_movement`, `separation_gain`,
  `received`, `lead_to_shot`, con controllo per sottotipo/fase; robustezza della calibrazione su Euro
  2024 se il regolamento lo permette.
- *Demo per l'hackathon:* video di 30–45 s broadcast → follow → POV sul clip Jelacic/Williams, con
  radar e curva del ghost error fino allo "scatto"; viewer online con 4–6 clip scelti insieme.

---

## 2026-10-04 — Avvio: port del modello della tesi, ghost score, prima analisi

**Obiettivo**
Riprodurre il Track A della tesi (FoV → Kalman → awareness) sui dati body pose SkillCorner, provare
l'awareness come **distanza tra posizione reale e posizione creduta (il "ghost")**, e costruire una
prima parte grafica (campetto 2D con ghost + viewer 3D con POV del difensore).

**Fatto**
- Esplorata la repo condivisa e la tesi; creata la cartella **`AWARENESS/`** al top level, speculare a
  `DRS/` di Beppe. Nessun file fuori da questa cartella è stato modificato.
- Scaricate da Hugging Face le due partite con body pose (1925299 Brisbane–Perth, 1996435
  Sydney–Adelaide) in `data_local/`; ambiente Python 3.12 in `.venv/` (con il 3.14 di Homebrew molte
  librerie non hanno ancora wheel).
- Scritta la libreria `ghost/` e gli script `01`–`06`; test `tests/check_kalman_port.py`.
- Prima versione del viewer 3D e delle GIF 2D (il viewer è stato poi rifatto il 2026-10-05).

**Situazione della repo condivisa**
- `master` = fork dei dati/tutorial SkillCorner + `ROADMAP.md` (bozza Beppe/Filippo).
- Branch remoti `DRS_Beppe` e `DRS_beppe` (stesso contenuto) con la cartella `DRS/` di Beppe
  (`nearest_defender.py`, `build_full_dataset.py`, `context_features.py`, …).
- Il tracking 10 fps (`*_tracking_extrapolated.jsonl`) è in **Git LFS** e sul Mac non c'è `git lfs`: i
  file sono puntatori da 133 byte. **Non serve**: il file body pose contiene già x/y di tutti i giocatori e
  della palla a 25 fps. Se servisse: `brew install git-lfs && git lfs pull`.
- Tempi su M-series: parsing di una partita 3.3 GB → cache parquet ≈ 4 min (una volta sola);
  orientamento ≈ 7 s; una finestra di 13 s ≈ 0.5 s; tutti i ~450 run di una partita ≈ 4 min.

### Scelte metodologiche

#### Dati
| Scelta | Motivazione |
|---|---|
| Unica fonte: file **body pose 25 fps** | Ha x/y di tutti + palla + possesso + 29 joint. dt = 0.04 s è identico a Hawk-Eye → parametri della tesi riusabili senza riscalare. |
| Filtro "in campo" con `start_time/end_time` di `match.json` vs timestamp del record | Il file pose elenca **tutta la rosa (32)** con x/y anche per panchinari/sostituiti (valori stantii). Dopo il filtro: esattamente 22 giocatori per frame su tutta la partita ✔. Il timestamp è il cronometro di gara (il 2° tempo parte da 45:00), stesso formato di `match.json`. |
| Coordinate | Verificato sul sample: `midHip − (x, y)` ha mediana ≈ 5 cm → i joint sono nello stesso sistema del campo. `z` si comporta come un'altezza (naso ≈ 1.6 m, piedi ≈ 0.15 m) anche se il README dice "solo relativo al centroide": la uso **solo per disegnare** (rebase: piede più basso a 3 cm da terra). |
| Cache compatta | Tengo solo naso, orecchie, spalle (+ errori p90) per ogni player-frame (3.4 M righe). Gli scheletri completi si estraggono al volo solo per le finestre da visualizzare. |
| Frame 10 fps ↔ 25 fps | `pose_frame = 2.5 × tracking_frame` (eventi dinamici e phases sono a 10 fps). |

**Copertura pose (limite principale vs Hawk-Eye):** joint presenti nel **46%** dei player-frame
(Hawk-Eye: ~100%). Dipende dalla distanza dalla palla (camera broadcast):

| distanza dalla palla | head angle da pose |
|---|---|
| 0–10 m | 71% |
| 10–20 m | 62% |
| 20–30 m | 44% |
| 30–50 m | 23% |
| > 50 m | 2% |

Per il deep-dive su situazioni vicine alla palla va bene; per metriche "tutta la partita" no.

#### Orientamento della testa (proxy dello sguardo)
1. **Formula identica alla tesi:** `θ = atan2(naso − media(orecchie))` sul piano del campo.
2. **Gating:** lunghezza planare naso–orecchie in [5, 25] cm e errore p90 dei joint testa ≤ 20 cm.
3. **Smoothing circolare:** mediana mobile centrata (5 frame = 0.2 s, elimina i flip fronte/retro
   di un frame) + media mobile sul vettore unitario. Jitter grezzo: mediana 3.4°/frame (≈ 85°/s di rumore).
4. **Buchi ≤ 1 s:** interpolazione lineare sull'arco più corto (`head_source = 'interp'`, ma sono solo lo
   0.5% dei frame: quando la pose manca, manca a lungo perché il giocatore è fuori inquadratura).
5. **Buchi lunghi: prior "guarda la palla"** (`head_source = 'ball_prior'`, 54% dei frame totali).
   È un'ipotesi forte → tutti gli score vanno filtrati su un difensore focale con
   `head_source ∈ {pose, interp}` (nei run uso `head_ok ≥ 60%`). Nel viewer il cono è più trasparente
   quando l'angolo è il prior.
6. **Validazione della convenzione** (importante: un segno sbagliato renderebbe tutto spazzatura):
   con angolo da pose, la testa è entro ±60° dalla direzione della palla nell'**88%** dei casi
   (spalle 78%, direzione di corsa 84% sopra 4 m/s). ✔
7. **Spalle per l'occlusione:** quando mancano, spalle sintetiche larghe 40 cm perpendicolari alla testa
   (stile Bekkers, che usa larghezza fissa).
8. **Velocità:** differenze su posizione smussata (7 frame), divisa per il passo di frame reale (salti a
   inizio tempo/sostituzioni) e clippata a 12 m/s (c'erano glitch fino a 339 m/s).

#### FoV e occlusione
Copiato senza modifiche dalla tesi (Bekkers SSAC26): cono ±60°, gaussiane radiale/angolare dipendenti
dalla velocità, ombre di occlusione dalle spalle, floor di prossimità 0.7 sotto 0.5 m. Nessun parametro
toccato, così i numeri restano confrontabili con Euro 2024.

#### Filtro di Kalman — cosa è uguale e cosa è cambiato
Uguale: stato [px, py, vx, vy], osservazione = posizione vera, **R adattiva = σ_obs²/w_ij**, hard skip
se w ≤ 0.01, versione batched su tutte le coppie, warm-up di 10 s prima della finestra di interesse
(come le cross window della tesi). Test: `tests/check_kalman_port.py` → identico alla tesi.

Tre cambiamenti, tutti opzionali (i default della tesi restano disponibili e vengono calcolati in
parallelo come colonne `trP_thesis`, `A_cov_thesis` per la riproduzione):

**(a) Motion prior calibrato sui dati (il cambiamento più importante).**
- *Problema osservato:* con il CV della tesi, un attaccante non visto per molti secondi fa "scappare" il
  ghost a velocità costante (errori fino a 86 m, ghost fuori dal campo). Per la tesi era irrilevante
  (l'awareness usava solo P), per il ghost no.
- *Primo tentativo scartato:* decadimento della velocità con la Q della tesi → P diventa ancora più
  stretta (troppi false belief) e il tracking in vista peggiora. Invece di tarare a mano:
- *Approccio:* il belief è un osservatore **normativo** → il suo modello di moto deve essere quello che
  predice meglio dove vanno davvero i giocatori quando smetti di guardarli. Script 02: ~6000 segmenti
  reali (solo frame `is_detected`, perché le posizioni estrapolate da SkillCorner sono già output di un
  modello e sarebbero troppo lisce), osservo 3 s con w = 1, poi w = 0 per 5 s, valuto a 0.5/1/2/3/5 s
  errore mediano, **NLL** (proper scoring rule) e **copertura dell'ellisse 95%**.
- *Griglia:* CV tesi (σ_acc 1–8), CV continuo, OU (τ 0.5–12 s × σ_v 1–3 m/s), discretizzazione esatta
  con Van Loan (`ghost/motion.py`).
- *Risultato (entrambe le partite):*

| prior | NLL medio | copertura 95% media | errore a 1 s | errore a 3 s |
|---|---|---|---|---|
| **OU τ = 4 s, σ_v = 2 m/s** (scelto) | **4.46** | **0.94** | 1.14 m | 4.45 m |
| OU τ = 3, σ_v = 2 | 4.47 | 0.95 | 1.17 | 4.46 |
| CV tesi σ_acc = 6 | 4.67 | 0.93 | 1.12 | 4.74 |
| **CV tesi σ_acc = 2** (default tesi) | 8.96 | **0.48** | 1.57 | 5.36 |

  Ottimo piatto tra τ = 3 e 6 s → scelta robusta. Figura: `outputs/calibration_motion_prior.png`.
- *Interpretazione:* τ = 4 s vuol dire che il ghost continua la corsa per qualche secondo e poi si ferma;
  σ_v = 2 m/s è la variabilità tipica della velocità di un giocatore.
- *Effetti:* errore del ghost **in vista** sceso da 0.55 a **0.18 m** (meno lag), false belief da 32% a
  **3.4%** (≈ il 5% nominale atteso se il modello è calibrato).
- *Nota per la tesi:* la Q della tesi `σ²[[dt⁴/4, dt³/2],[dt³/2, dt²]]` è la forma **DWNA** (discrete
  white-noise acceleration) di Bar-Shalom, non la CWNA come scritto nelle docstring. Entrambe legittime,
  è solo il nome.

**(b) Inizializzazione al primo avvistamento (`init_on='seen'`).** Nella tesi la credenza nasce al primo
frame in cui la coppia esiste, **dalla posizione vera, anche se il difensore non l'ha mai visto**. Per la
covarianza conta poco (P0 = 100 I), ma per il ghost crea credenze non giustificate da alcuna
osservazione (nei run c'erano ghost a 44 m "inventati"). Ora la credenza nasce alla prima osservazione
con w > w_min; prima lo stato è `never_seen` con A_ghost = A_cov = 0.

**(c)** Il filtro restituisce anche la velocità stimata (per orientare il ghost nel viewer).

#### Gli score
Tutti calcolati **post-hoc** dallo stesso output (come nella tesi: cambiare formula non richiede di
rifare il filtro).

| score | formula | lettura |
|---|---|---|
| `A_cov` (tesi) | exp(−tr P / 8 m²) | quanto è incerto il difensore |
| `ghost_err` (**nuovo**) | ‖x̂ − x‖ in metri | quanto è sbagliata la sua immagine mentale |
| `A_ghost` | exp(−(err / 2 m)²) | versione in (0, 1]; 2 m ≈ un passo + allungo |
| `mahal2` | eᵀ P⁻¹ e ~ χ²(2) sotto il modello | quanto è "sorprendente" la posizione vera per lui |
| `belief_state` | aware (err < 1.5 m) / uncertain (err ≥ 1.5, mahal2 ≤ 5.99) / **false_belief** (err ≥ 1.5, mahal2 > 5.99) / never_seen | quadranti |

**Perché il ghost aggiunge qualcosa rispetto alla tesi (punto da discutere):** nel Kalman, **P non
dipende da cosa fa l'attaccante**, solo da quando è stato visto (la sequenza di w). Quindi `A_cov` è, di
fatto, un FoV smussato nel tempo. Il ghost error invece dipende dall'**interazione** tra attenzione e
comportamento dell'attaccante: un attaccante non visto che corre dritto resta vicino al suo ghost (il
difensore lo può estrapolare), uno che cambia direzione/velocità mentre non è visto apre un gap → proprio
i movimenti alle spalle che interessano nel *defensive positioning*.

**Onestamente:** sulle coppie entro 25 m, ghost_err e A_cov hanno correlazione di Spearman −0.90 (entrambi
crescono col tempo non visto). Il valore aggiunto è nel **residuo**, ed è esattamente ciò che misura
`mahal2` / `false_belief`. Può anche aiutare a spiegare perché nella tesi A_cov non migliorava la GNN: era
quasi ridondante con informazione di FoV/geometria già presente.

**Aggregazione a livello difensore** (`scores.defender_level`): `max` (peggior punto cieco), `nearest`,
`dist_weighted` (default, peso exp(−d/15 m)). Per i run uso la **coppia focale**
(difensore più vicino al corridore a inizio run ↔ corridore).

#### Finestre e selezione delle situazioni (ponte con la DRS)
- Uso gli **`off_ball_run`** degli eventi dinamici SkillCorner (sottotipi: behind, run_ahead_of_the_ball,
  cross_receiver, overlap, underlap, pulling_half_space, pulling_wide, support, coming_short) = gli stessi
  "test standardizzati" della DRS di Beppe.
- Finestra = [inizio run, fine run + 1 s], warm-up 10 s, osservatori = squadra avversaria del corridore.
- Difensore focale = avversario più vicino al corridore all'inizio del run.
- Affidabilità: `head_ok ≥ 0.6` (≥ 60% dei frame del focale con testa da pose) e distanza iniziale < 12 m.
- Metriche per run: `seen_frac`, `ghost_err_max`, `ghost_err_end` (ghost error a fine run, cioè quando
  arriverebbe la palla), `false_belief_s`, `never_seen_s`, `A_cov_thesis_mean`, `A_ghost_mean`, più gli
  esiti SkillCorner (`targeted`, `received`, `dangerous`, `lead_to_shot`, `xthreat`).
- Output: `outputs/offball_runs_{match}.parquet` (897 run, 776 affidabili sulle due partite).

### Risultati
**Per sottotipo di run** (media sui run affidabili, 2 partite):

| sottotipo | n | % frame in cui il difensore vede il corridore | s in false belief | ghost err a fine run (m) |
|---|---|---|---|---|
| cross_receiver | 80 | **35%** | **0.97** | **7.8** |
| underlap | 29 | 41% | 0.79 | 3.2 |
| support | 140 | 42% | 0.63 | 5.4 |
| run_ahead_of_the_ball | 270 | 43% | 0.54 | 5.0 |
| overlap | 24 | 49% | 0.52 | 2.5 |
| behind | 51 | 53% | 0.54 | 3.2 |
| coming_short | 117 | 55% | 0.29 | 4.3 |
| pulling_wide | 50 | 57% | 0.12 | 3.0 |
| pulling_half_space | 15 | 79% | 0.05 | 1.7 |

**Dangerous vs non dangerous** (Mann–Whitney, 2 partite pooled):

| metrica | dangerous | non dangerous | p |
|---|---|---|---|
| seen_frac | 0.40 | 0.50 | 0.001 |
| ghost_err_end | 6.1 m | 4.1 m | 0.004 |
| A_ghost_mean | 0.48 | 0.56 | 0.001 |
| A_cov_thesis_mean | 0.65 | 0.70 | 0.006 |

**Ma, controllando per sottotipo** (logit `dangerous ~ sottotipo + z(score)`): A_ghost coef −0.13
(z = −1.29), A_cov coef −0.11 (z = −1.13). Direzione coerente in quasi tutti i sottotipi, **non
significativo**. Caveat aggiuntivi: run non indipendenti (stessi giocatori, finestre sovrapposte), test
multipli, `dangerous` è un'etichetta SkillCorner a livello di run, solo 2 partite.
→ Da presentare come **spiegazione/visualizzazione**, coerente con la decisione già presa in ROADMAP
("risultato nullo della tesi → presentarla come spiegazione, non feature predittiva").

**Clip di esempio** (match 1925299, scelti automaticamente = più tempo in false belief):
- 46' cross receiver D. Williams — K. Jelacic (#23) guarda la palla/il cross, Williams fa una lunga corsa
  alle sue spalle non visto (13% dei frame, 7.4 s in false belief); quando si gira, il ghost "scatta"
  sulla posizione vera. **È il clip migliore per la demo.**
- 4' coming short N. Pennington, 74' run ahead J. Hingert, 46' cross receiver A. Bugarija.

**Sintesi della giornata**
1. La pipeline della tesi gira sui dati SkillCorner: FoV e Kalman copiati 1:1 (il filtro portato è
   identico a `run_filter` della tesi su 20 coppie × 300 frame).
2. Nuovo score "ghost": `ghost_err = ‖x̂ − x‖` in metri, `A_ghost = exp(−(err/2 m)²)`, e Mahalanobis per
   distinguere "so di non sapere" da "credo di sapere ma sbaglio" (false belief).
3. Il motion prior della tesi (CV, σ_acc = 2) è molto sovra-confidente (copertura dell'ellisse 95% al
   48%); il prior calibrato OU (τ = 4 s, σ_v = 2 m/s) è ben calibrato (94%) e dimezza la NLL.
4. Sugli off-ball run il segnale grezzo c'è, ma controllando per sottotipo non è significativo; i cross
   sono il caso più evidente di *ball watching*.

### Grafica 2D
**2D** (`ghost/viz2d.py`, GIF e PNG in `outputs/media/clips/`, non tracciata): pallini pieni = giocatori
reali; cerchio tratteggiato = ghost (dove il difensore focale crede sia l'attaccante); linea ghost→reale
colorata per errore (verde < 1.5 m < ambra < 4 m < rosso); ellisse 95% solo per la coppia focale; cono
±60° del focale; pannelli laterali con w_ij, ghost error, A_cov (tesi) vs A_ghost. GIF a 12.5 fps (manca
ffmpeg per gli mp4: `brew install ffmpeg` se servono video).

### Limiti e cautele
1. **Il ghost dipende dall'osservazione "vera".** Come nella tesi, l'osservazione che entra nel filtro è la
   posizione vera di tracking. Il ghost misura quindi un osservatore ideale con attenzione limitata, non la
   percezione reale (niente eye-tracking; la testa è un proxy dello sguardo). Va detto chiaramente nel paper.
2. **Posizioni estrapolate.** Quando l'attaccante non è detected (fuori camera), anche la "verità" è una
   stima SkillCorner. Ogni riga ha `target_detected`: per analisi rigorose filtrare lì.
3. **σ_obs = 1 m** (tesi) è una scelta percettiva, non la precisione del tracking. Non ricalibrata: si può
   fare con lo stesso schema dello script 02.
4. **τ dello score A_cov (8 m²)** era stato scelto nella tesi per il prior CV. Con il prior OU la P cresce
   più in fretta → `A_cov` (calibrato) scende prima. Per confronti con la tesi usare `A_cov_thesis`.
5. **Warm-up di 10 s:** credenze più vecchie di 10 s vengono tagliate; per finestre vicine all'inizio di un
   tempo il warm-up è più corto (clamp al periodo).
6. `tests/check_kalman_port.py` importa il codice della tesi da un path assoluto del Mac: va bene in
   locale, non in CI.

### Come riprodurre
```bash
cd AWARENESS
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import sys; sys.path.insert(0,'..'); from src.data.pose_loading import download_match; [download_match(m, dest_dir='data_local') for m in (1925299, 1996435)]"
.venv/bin/python scripts/01_prepare_match.py 1925299 && .venv/bin/python scripts/01_prepare_match.py 1996435
.venv/bin/python scripts/02_calibrate_motion_prior.py 1925299 1996435
.venv/bin/python scripts/03_offball_runs.py 1925299 && .venv/bin/python scripts/03_offball_runs.py 1996435
.venv/bin/python scripts/04_make_demo.py 1925299 4
.venv/bin/python tests/check_kalman_port.py
```

Screenshot headless del viewer (opzionale, usa Brave già installato, nessun download di browser):
`.venv/bin/python scripts/06_screenshot_viewer.py [indice_clip] [frazione_del_clip]`
→ `outputs/media/screenshots/clip{i}_{broadcast,top,follow,pov}.png`.
