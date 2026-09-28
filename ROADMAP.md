# Roadmap — DRS × Awareness

> Documento di lavoro condiviso tra per SkillCorner × PySport Analytics Cup 2.
> Tema: **Defensive Positioning** · Deadline submission: **18 dicembre 2026** · Finale europea: Parigi, 17 febbraio 2027
> Stato: bozza da discutere, POSSIAMO CAMBIARE MAIN METHODOLOGY

---

## 1. Da dove partiamo

Dalla prima chiacchierata sono emerse due linee di lavoro che si incastrano bene.

| | **DRS — Defensive Response Signature** (Beppe) | **Modello cognitivo / awareness** (Filippo) |
|---|---|---|
| Idea | Usare le tassonomie di attaccanti di SkillCorner (10 subtype di `off_ball_run`, 3 tipi di `passing_option`) come "test standardizzati" per profilare cosa una difesa/un difensore gestisce bene o male | Modellare cosa il difensore *sa* della situazione: campo visivo (FoV) + occlusioni + Kalman filter come belief state sugli avversari |
| Domanda | **Quando** e **contro cosa** la difesa va in difficoltà? | **Perché** un difensore sbaglia: non vedeva l'avversario? Postura, tempi di reazione? |
| Dati | 20 match XY tracking + eventi dinamici (A-League 24/25) | 2 match con body pose 3D; tesi su Hawk-Eye Euro 2024 |
| Output | Profilo situazionale per squadra e per ruolo | Replay con coni visivi, leaderboard di attenzione |
| Punto di forza | Aggregabile per squadra, spendibile per lo scouting | Interpretabilità sul singolo, forte impatto visivo |

**Convergenza in una frase:** DRS individua i momenti di pericolosità e le situazioni deboli; l'awareness li spiega a livello di singolo difensore.

## 2. Cosa ci siamo detti (sintesi delle idee)

**Su cosa costruire**
- Nessuna metrica aggregata "media sul campionato": interessa il **punto debole di una singola squadra** (mia squadra o avversario), quindi profilo per squadra.
- Molto lavoro esistente sulla difesa è aggregato; **poche metriche individuali** → lì c'è valore aggiunto.
- Originalità e **spendibilità per un club**, non ricerca accademica pura. Lezione dalle submission dell'anno scorso: idea semplice, concetto preso da un altro campo, demo chiara.
- **Visual first**: replay 2D (eventualmente 3D) che mostra la situazione e come cambia l'indice di pericolo.

**Come collegare le due parti**
- L'indice di pericolosità (DRS) fa da **filtro**: individua le N situazioni più pericolose di una partita.
- Sulle sole sequenze filtrate si esegue l'analisi 3D/awareness (il 3D è pesante, si usa solo per **deep dive su sequenze brevi**).
- Le fasi di gioco del dataset (high/mid/low block, transizioni) servono da filtro/stratificazione iniziale.
- Il tracking serve a trovare ciò che l'evento non vede: spazi che si aprono improvvisamente anche senza passaggio successivo.

**Idee di analisi emerse**
- Matrice **subtype × fase difensiva** per squadra (es. corse in profondità contro mid block).
- Stratificare la pericolosità per situazione: sovrapposizione, inserimento alle spalle, ecc.
- 
- 
- 
- 

**Casi d'uso per un club**
1. Leaderboard dei difensori più/meno attenti in specifiche situazioni.
2. Analisi di un gol subito: non solo *chi* ha sbagliato ma *perché* (tempi, inserimento non visto).
3. Scouting avversario: punti deboli specifici per squadra.

## 3. Decisioni prese

- **A + B portanti**: DRS (profilo situazionale) + attribuzione individuale dal tracking (nearest defender / marking / leave-one-out).
- **C come showcase qualitativo**: awareness 3D sui 2 match con body pose, come sezione "deep dive" nel paper. I dati 3D sono troppo pochi per essere un pilastro.
- Il modello resta **leggero e interpretabile** (statistica / regole, non deep learning), con un modello predittivo solo in fase finale come validazione.
- Sforzo contenuto: ognuno ci dedica tempo limitato, deve essere qualcosa da cui impariamo entrambi.

## 4. Roadmap

### Fase 0 — Allineamento (fino a metà ottobre)
- [ ] **Filippo**: demo del Kalman filter sui dati 3D + piccola presentazione per il prossimo incontro (belief state, quanto si discosta la posizione stimata da quella reale, valori estraibili).
- [ ] **Filippo**: verificare il regolamento sull'uso di dati esterni (i 50 match 3D Hawk-Eye) — anche solo come validazione.
- [ ] **Beppe**: chiudere il dataset unificato Layer 3 (8.877 run valide su 20 match) e condividerlo.
- [ ] **Entrambi**: scegliere la definizione operativa di "pericolosità" (vedi §6) e le prime 2–3 domande a cui vogliamo rispondere.

### Fase 1 


### Fase 2

### Fase 3 

### Fase 4 

### Fase 5 

## 5. Possibili sviluppi (oltre la deadline)


## 6. Domande aperte

1. **Definizione di pericolosità**: esito dell'azione (ricevuta / tiro / gol) vs. probabilità predetta da un modello?
2. Possiamo usare i dati Hawk-Eye Euro 2024 nella submission, anche solo per validazione? *(da verificare: Filippo)*
3. Come si "trasferisce" il concetto di awareness dai dati Hawk-Eye ai 2 match body pose di SkillCorner (qualità della pose, frequenza)?
4. Su quale unità aggreghiamo il profilo: squadra, ruolo/zona, giocatore? Con 20 match il campione per giocatore è piccolo.
5. Come gestire il risultato nullo della tesi (awareness non migliora la predizione di danger)? Proposta: presentarla come **spiegazione**, non come feature predittiva.
6. Come dividere il lavoro dell'ultima settimana (packaging vs. video vs. paper)?

## 7. Ruoli proposti


## 8. Prossimo incontro

- Filippo porta la **demo Kalman** con mini-presentazione.
- Beppe porta lo stato del **dataset Layer 3** e il primo profilo DRS per squadra.
- Decidiamo insieme la definizione di pericolosità e chiudiamo le domande aperte 1–2.

---

*Repo: `analytics_cup_2.0_drs` · branch di lavoro: `master`.*
