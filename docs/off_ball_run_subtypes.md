# Off-ball run subtypes — guida di riferimento

> Documento di contesto per **Analytics Cup PySport (DRS × Awareness)**.
> Scopo: avere in ogni chat del progetto la stessa definizione di ciascun subtype di `off_ball_run`, per non confondersi a livello tattico e pratico.
> Dati: Layer 3 (`layer3_all_matches.csv`), 8.877 run valide su 20 match A-League 24/25.
> Definizioni ufficiali: video SkillCorner "Game Intelligence – Off Ball Run Types" (screenshot e didascalie).

---

## 0. Come leggere questo documento

Ogni informazione ha uno di tre livelli di affidabilità. **Non mescolarli** quando si cita il documento in un'altra chat.

| Livello | Cosa contiene | Affidabilità |
|---|---|---|
| **[UFFICIALE]** | Definizione SkillCorner, come scritta nella didascalia del video | Alta, è la fonte |
| **[DATO]** | Statistiche calcolate dal Layer 3 | Alta come calcolo, ma la semantica di alcune colonne va confermata (vedi §6) |
| **[IPOTESI]** | Lettura difensiva e interpretazioni | Da validare, non usare come fatto |

Le corse sono corse **della squadra in possesso**. Per la difesa (tema della Cup) ogni subtype è la "situazione" a cui il difensore deve rispondere.

---

## 1. Lo schema: cosa distingue un subtype dall'altro [UFFICIALE]

SkillCorner divide il campo in **3 canali longitudinali**: `CENTER`, `HALF SPACE`, `WIDE` (visibili negli overlay dei frame). Le 10 definizioni si distinguono per quattro riferimenti:

1. **Rispetto al portatore di palla:** parte da dietro / da davanti / va verso di lui.
2. **Canale:** dove parte e dove finisce (center, half space, wide).
3. **Rispetto all'ultima linea difensiva avversaria:** non la supera / va oltre.
4. **Zona:** terzo offensivo, area di rigore.

| Subtype (CSV) | Nome SkillCorner | Riferimento principale |
|---|---|---|
| `dropping_off` | Dropping Off | verso la **porta della propria squadra** |
| `coming_short` | Coming Short | verso il **portatore** |
| `pulling_wide` | Pulling Wide | canale: center/half space → **wide** |
| `pulling_half_space` | Pulling Half Space | canale: center → **half space** (terzo offensivo) |
| `support` | Support | parte da **dietro** il portatore |
| `run_ahead_of_the_ball` | Run Ahead | parte **davanti** al portatore, **non** supera l'ultima linea |
| `overlap` | Overlap | da dietro, finisce davanti e **fuori** |
| `underlap` | Underlap | da dietro, finisce davanti e **dentro** |
| `behind` | Run in Behind | **oltre** l'ultima linea difensiva |
| `cross_receiver` | Cross Receiver | verso l'**area di rigore** |

Attenzione al nome nel CSV: `behind` = "Run in Behind", `run_ahead_of_the_ball` = "Run Ahead".

---

## 2. Schede per subtype

Le statistiche sono sul Layer 3 (n = numero di run, durata media in secondi assumendo 10 Hz, targeted/received/shot in % delle run del subtype, sep = separation media a inizio corsa).

### 2.1 `dropping_off` — Dropping Off
- **[UFFICIALE]** *"A run towards the team in possession's goal to open a passing angle."* Corsa verso la porta della squadra in possesso per aprire un angolo di passaggio.
- **Nei frame:** due giocatori cerchiati in rettangoli verdi, collegati da tratteggi al portatore.
- **[IPOTESI] Lettura difensiva:** il runner si allontana dalla porta avversaria. Il difensore sceglie se seguirlo (rompendo la linea) o lasciarlo. Il rischio è lo spazio alle spalle, non la profondità.
- **[DATO]** n = 1.010 · durata 2,1 s · targeted 39,1% · received 38,1% · shot 2,2% · sep 9,4 (la più alta) · gain **+0,5**.
- **Non confondere con:** `coming_short`.

### 2.2 `coming_short` — Coming Short
- **[UFFICIALE]** *"A run towards the player in possession to receive a short pass."* Corsa verso il portatore per ricevere un passaggio corto.
- **Nei frame:** linea verde/rossa che termina in un cerchio sul punto di ricezione, tratteggio bianco verso i compagni.
- **[IPOTESI] Lettura difensiva:** il runner viene incontro alla palla; il difensore decide se accorciare o tenere la linea.
- **[DATO]** n = 1.085 · durata 2,1 s · targeted 34,5% · received 32,7% · shot 2,6% · sep 5,2 · gain −0,4.
- **Non confondere con:** `dropping_off`. Cambia il **riferimento** della direzione: portatore (`coming_short`) contro porta della propria squadra (`dropping_off`). I due movimenti possono coincidere quando il portatore è più indietro del runner.

### 2.3 `pulling_wide` — Pulling Wide
- **[UFFICIALE]** *"A run starting in the center or half space channel and ending in the wide channel (with a location wider than the player in possession)."*
- **Nei frame:** curva rossa che parte dal half space e termina nel canale WIDE.
- **[IPOTESI] Lettura difensiva:** il runner allarga il campo e trascina un difensore, oppure resta libero sulla fascia. Se la difesa segue, si apre spazio centrale.
- **[DATO]** n = 564 · durata 2,4 s · targeted 43,6% · received 39,4% · shot 3,7% · sep 6,8 · gain −0,3.
- **Non confondere con:** `pulling_half_space` (arrivo diverso), `overlap` (parte da dietro il portatore).

### 2.4 `pulling_half_space` — Pulling Half Space
- **[UFFICIALE]** *"A run occurring in the attacking third, that starts in the center channel and ends in the half space channel."*
- **Nei frame:** linea che parte dal canale CENTER e termina nel HALF SPACE.
- **[IPOTESI] Lettura difensiva:** il runner esce dal centro verso la zona tra difensore centrale e terzino. Il rischio è il passaggio di consegne tra reparti.
- **[DATO]** n = 252 · durata 2,1 s · targeted 26,2% · received 22,6% · shot 5,2% · sep 4,3 · gain −0,3. Campione piccolo.
- **Non confondere con:** `pulling_wide`. **Attenzione:** solo questa definizione contiene il vincolo "terzo offensivo". Non sappiamo se valga anche per le altre (vedi §6).

### 2.5 `support` — Support
- **[UFFICIALE]** *"A run originating behind the player in possession trying to engage in offensive play."*
- **Nei frame:** più giocatori cerchiati dietro la palla; compare una lunga linea diagonale tratteggiata (significato da verificare nel video).
- **[IPOTESI] Lettura difensiva:** il runner non è la minaccia più vicina alla linea; conta quanti giocatori si aggiungono all'azione e se restano agganciati.
- **[DATO]** n = 1.455 · durata 2,5 s · targeted 26,9% · received 23,0% · shot 15,5% · sep 5,5 · gain **+0,3**.
- **Non confondere con:** `overlap`/`underlap` (anch'essi da dietro, ma con arrivo davanti e fuori/dentro). Il report CIES aggiunge "durante una transizione offensiva": è una fonte terza, **non** nella definizione SkillCorner.

### 2.6 `run_ahead_of_the_ball` — Run Ahead
- **[UFFICIALE]** *"A run originating ahead of the player in possession, running towards the opponents' goal but not going beyond the last defensive line."*
- **Nei frame:** lunga linea tratteggiata del colore della squadra in possesso, runner davanti al portatore e prima di quella linea. Probabile ultima linea difensiva, **da confermare**.
- **[IPOTESI] Lettura difensiva:** il runner minaccia la profondità ma resta davanti alla linea. La scelta è alzarsi con la palla o scendere con la corsa.
- **[DATO]** n = 2.498 (28% del totale, il più frequente) · durata 2,4 s · targeted 33,0% · received 25,1% · shot 12,0% · sep 5,5 · gain −0,5.
- **Non confondere con:** `behind` (stessa direzione, ma `behind` supera la linea), `support` (parte da dietro).

### 2.7 `overlap` — Overlap
- **[UFFICIALE]** *"A run in the wide or half space channels, originating behind the player in possession and ending ahead and outside of the player in possession."*
- **Nei frame:** due giocatori cerchiati vicini sulla fascia; non è chiaro quale sia runner e quale portatore.
- **[IPOTESI] Lettura difensiva:** il runner passa all'esterno e crea un 2 contro 1 sulla fascia. Il terzino segue o l'esterno scala.
- **[DATO]** n = 270 · durata **3,6 s** (la più lunga) · targeted 40,0% · received 36,3% · shot 19,6% · sep 6,6 · gain **−1,6** (il calo maggiore). Campione piccolo.
- **Non confondere con:** `underlap` (stessa definizione ma lato interno).

### 2.8 `underlap` — Underlap
- **[UFFICIALE]** *"A run in the wide or half space channels, originating behind the player in possession and ending ahead and inside of the player in possession."*
- **Nei frame:** due giocatori cerchiati vicino all'area; linea azzurra che esce da uno dei due.
- **[IPOTESI] Lettura difensiva:** il runner passa all'interno, verso la zona tra terzino e difensore centrale.
- **[DATO]** n = 226 (il meno frequente) · durata 2,8 s · targeted 30,5% · received 22,1% · shot 14,6% · sep **3,5** (la più bassa) · gain −0,7. Campione piccolo.
- **Non confondere con:** `overlap`. Differiscono solo per "fuori" contro "dentro". **Nota:** la definizione SkillCorner indica wide o half space, **non** il canale centrale (come invece scrive Driblab).

### 2.9 `behind` — Run in Behind
- **[UFFICIALE]** *"A run into the space behind the opponents' last defensive line."*
- **Nei frame:** difensori cerchiati e linee della squadra in possesso che vanno verso lo spazio dietro di loro.
- **[IPOTESI] Lettura difensiva:** il runner minaccia la profondità. La difesa sceglie se scendere, tenere la linea (fuorigioco) o scalare. Il subtype in cui le scelte della linea contano di più.
- **[DATO]** n = 648 · durata 2,8 s · targeted **44,3%** (il più alto) · received 21,5% · shot 20,1% · sep 4,1 · gain −0,9.
- **Non confondere con:** `run_ahead_of_the_ball`. Il confine è l'ultima linea difensiva, quindi i casi vicini alla linea possono confondersi.

### 2.10 `cross_receiver` — Cross Receiver
- **[UFFICIALE]** *"A run towards the penalty area to receive a possible cross."* Corsa verso l'area di rigore per ricevere un possibile cross.
- **Nei frame:** area di rigore evidenziata in giallo; il runner è cerchiato al suo margine con una linea di corsa.
- **[IPOTESI] Lettura difensiva:** riguarda il posizionamento in area su cross e il marcare chi entra. La corsa esiste a prescindere dal fatto che il cross arrivi ("possible cross").
- **[DATO]** n = 869 · durata 3,1 s · targeted 32,7% · received **12,8%** · shot **28,4%** (il più alto) · sep 4,4 · gain −1,3 (secondo calo maggiore).
- **Non confondere con:** `behind` e `run_ahead_of_the_ball` quando la corsa va verso l'area ma non c'è cross.

---

## 3. Coppie facili da confondere

| Coppia | Cosa le distingue [UFFICIALE] |
|---|---|
| `dropping_off` ↔ `coming_short` | Riferimento: porta della propria squadra contro portatore |
| `pulling_wide` ↔ `pulling_half_space` | Canale di arrivo (wide contro half space); partenza da center o half space contro solo center |
| `overlap` ↔ `underlap` | Arrivo davanti e **fuori** contro davanti e **dentro** |
| `overlap` / `underlap` ↔ `support` | Tutti partono da dietro il portatore; overlap e underlap hanno anche un arrivo davanti e un canale precisi |
| `run_ahead_of_the_ball` ↔ `behind` | Non supera contro supera l'ultima linea difensiva |
| `run_ahead_of_the_ball` ↔ `support` | Parte davanti contro parte da dietro il portatore |

---

## 4. Statistiche Layer 3 a confronto [DATO]

Fonte: `layer3_all_matches.csv`, 8.877 run, 20 match. Durata = (`frame_end` − `frame_start`) / 10, assumendo 10 Hz. Intervallo = Wilson 95% sulla % shot.

| Subtype | n | % tot | Durata (s) | Targeted % | Received % | Shot % [IC 95%] | Sep. inizio | Sep. gain |
|---|---|---|---|---|---|---|---|---|
| `run_ahead_of_the_ball` | 2.498 | 28,1 | 2,44 | 33,0 | 25,1 | 12,0 [10,8–13,4] | 5,47 | −0,49 |
| `support` | 1.455 | 16,4 | 2,51 | 26,9 | 23,0 | 15,5 [13,7–17,4] | 5,49 | +0,27 |
| `coming_short` | 1.085 | 12,2 | 2,07 | 34,5 | 32,7 | 2,6 [1,8–3,7] | 5,20 | −0,40 |
| `dropping_off` | 1.010 | 11,4 | 2,06 | 39,1 | 38,1 | 2,2 [1,4–3,3] | 9,36 | +0,52 |
| `cross_receiver` | 869 | 9,8 | 3,13 | 32,7 | 12,8 | 28,4 [25,5–31,5] | 4,37 | −1,25 |
| `behind` | 648 | 7,3 | 2,78 | 44,3 | 21,5 | 20,1 [17,2–23,3] | 4,05 | −0,89 |
| `pulling_wide` | 564 | 6,4 | 2,41 | 43,6 | 39,4 | 3,7 [2,5–5,6] | 6,78 | −0,31 |
| `overlap` | 270 | 3,0 | 3,58 | 40,0 | 36,3 | 19,6 [15,3–24,8] | 6,61 | −1,56 |
| `pulling_half_space` | 252 | 2,8 | 2,06 | 26,2 | 22,6 | 5,2 [3,0–8,6] | 4,27 | −0,28 |
| `underlap` | 226 | 2,6 | 2,80 | 30,5 | 22,1 | 14,6 [10,6–19,8] | 3,54 | −0,68 |

Osservazioni sui dati (non spiegazioni):
- In tutte le 8.877 run `received` implica `targeted` (0 eccezioni): `received` è un sottoinsieme di `targeted`.
- Il rapporto received/targeted è ~97% in `dropping_off` e ~48% in `behind`: nei subtype verso la palla il passaggio arriva quasi sempre, in profondità spesso no.
- `shot` separa due gruppi: ~2–5% (`dropping_off`, `coming_short`, `pulling_*`) e ~12–28% (gli altri).
- Solo `dropping_off` e `support` hanno separation in crescita; `overlap` e `cross_receiver` hanno il calo più forte.

---

## 5. Come usarlo nelle altre chat

- Citare sempre il nome del CSV (`behind`, non "run in behind") per evitare ambiguità.
- Quando si usa una definizione, dire se è **[UFFICIALE]** o **[IPOTESI]**.
- Non confrontare tra loro i subtype con n < 300 (`overlap`, `underlap`, `pulling_half_space`) senza intervalli di confidenza.
- Non leggere `shot %` come "pericolosità del subtype" finché non è chiaro come è costruito `lead_to_shot` (§6).

---

## 6. Punti aperti da verificare

1. **Definizione di `lead_to_shot`:** finestra temporale e catena di eventi. `cross_receiver` ha shot 28% ma received 13%, quindi il tiro non è necessariamente del runner. Può dipendere dalla zona della corsa più che dal tipo di movimento (`support` 15,5%, `dropping_off` 2,2%).
2. **Definizione di `separation` e `distance`:** nel CSV hanno la stessa media a inizio corsa (5,71). Probabilmente distanza dal difensore assegnato, da confermare nel codice del Layer 3 (nearest defender o marking assignment).
3. **Vincolo "terzo offensivo":** compare solo in `pulling_half_space`. Se vale solo per questo subtype, il confronto di frequenza con `pulling_wide` (564 contro 252) non è diretto.
4. **Linea tratteggiata lunga nei frame di `support` e `run_ahead_of_the_ball`:** ipotesi che sia l'ultima linea difensiva, non confermata. In `behind` la linea è mostrata come difensori cerchiati.
5. **Priorità tra definizioni:** le definizioni non sono mutuamente esclusive (es. `support` e `overlap` partono entrambi da dietro). Ogni run ha un solo subtype: non sappiamo come SkillCorner risolva i casi doppi.
6. **Confine `run_ahead_of_the_ball` / `behind`:** dipende da come viene individuata l'ultima linea in ogni frame.
7. **Fonti terze** (CIES Football Observatory, Driblab): descrivono i subtype in modo leggermente diverso (es. "transizione" per `support`, "canale centrale" per `underlap`). Vanno usate come contesto, non per interpretare i dati.
8. **Durata:** calcolata con 10 Hz assunto; da confermare con la frequenza reale del tracking.
