# Registro delle decisioni — DRS(t)

Log append-only. Una voce per decisione: **cosa**, **perché**, **i numeri che
l'hanno motivata**, **cosa la farebbe cambiare idea**.

Diverso da `ROADMAP.md` (piano condiviso con Filippo) e dal README di
`DRS/DRS_t/` (documentazione del modello come è oggi). Questo tiene traccia
anche delle strade **scartate**, che il README non riporta e che sono
materiale diretto per la sezione metodologica del paper.

Voci nuove in fondo. La tabella qui sotto è l'unica parte che si riscrive.

---

## Stato corrente

| voce | valore | deciso in |
|---|---|---|
| prior | geometrico `exp(-d/25)` sulla **partenza** | D-03, D-04 |
| `geo_use_endpoint` | False | D-05 |
| `geo_ball_gate` | False | D-06 |
| anisotropia angolare (verso la porta) | scartata | D-12, D-13 |
| fattore `T_r` palla-corridore | in prova, scala 20 m | D-14 |
| anisotropia direzionale | esaurita sui dati XY | D-15, D-16 |
| containment nello stress | `separation_gain` | D-07 |
| containment appreso | asse separato, non fattore | D-08 |
| curve containment | off | D-09 |
| struttura | enriched + tracking XY | — |
| validazione | dissociazione **e** split-half | D-10 |
| run di riferimento | `out/ref`, OOS, copertura tracking 100% | D-11 |
| copertura della curva | 21.4% per squadra (non 40.1%) | D-20 |
| profilo di squadra | `out/ref_learned` (containment appreso) | D-17 |

Numeri di riferimento (`out/ref`, 8877 corse, 20 match):

```
Dissociazione (OOS)   targeted     OR 0.993 [0.949, 1.039]  p=0.76
                      received     OR 1.019 [0.971, 1.070]  p=0.44
                      lead_to_shot OR 1.217 [1.155, 1.282]  p=2.1e-13

Split-half            metrica                    mediana   Sp.-Brown
                      deep_phase_share            +0.793     +0.885
                      stress_per_run              +0.724     +0.840
                      prior_mean                  +0.721     +0.838
                      drs_mean_all                +0.713     +0.832
                      n_runs                      +0.695     +0.820
                      stress_per_run_phase_adj    +0.534     +0.696
                      containment_failure_mean    +0.026     +0.050
                      containment_phase_adj       -0.074     -0.160
```

---

## D-01 · 7 ott 2026 · Strada C come impianto del framework

Gerarchia a stadi + grammatica discreta + awareness strutturale, senza
controfattuale rigoroso. Motivo: la deadline del 18 dicembre non regge un
controfattuale credibile, e il supporto non è garantito (positività).

Cambierebbe idea: solo uno slittamento della deadline.

## D-02 · 7 ott 2026 · DRS come curva temporale, non residuo

`DRS(t)` continuo; la decomposizione a 3 stadi diventa lettura degli spike,
non architettura del modello. L'awareness di Filippo entra come feature
frame-level, non solo come deep dive narrativo.

Motivo: lo stadio 1 `p(targeted | run)` modellava la **scelta del portatore**,
non la difesa — AUC 0.58 con baseline e feature player-level, e il target
`received` è miscela di quattro segnali incompatibili.

## D-03 · ott 2026 · Prior geometrico invece che dagli esiti

`danger = exp(-dist/scale)` dal punto di partenza. Non tocca gli esiti.

Le versioni precedenti derivavano dallo shot rate (per subtype, poi per
subtype × zona): circolare rispetto a `lead_to_shot`, da neutralizzare con
controlli nella logistica, al costo di segnale.

```
prior da esiti (subtype x third)   OR 1.165   p 1.3e-08
prior geometrico (scala 25 m)      OR 1.235   p 1.2e-15
```

In più, con prior geometrico in-sample e OOS coincidono (1.2167 vs 1.2171):
non c'è niente da stimare sul train. Variante ibrida geometrico × subtype
provata: OR 1.216, split-half 0.671 — nessuno dei due vantaggi pieni.

Cambierebbe idea: un prior esterno già validato e non derivato dai nostri
esiti (vedi Q-03).

## D-04 · ott 2026 · Scala del prior = 25 m

Criterio dichiarato: **massimizzare la split-half residualizzata**, non l'OR.

```
scala    OR      split-half   split-half residualizzato
12 m     1.179     0.581            0.551
16 m     1.206     0.582            0.558
20 m     1.223     0.583            0.562
25 m     1.235     0.576            0.563   <- scelta
30 m     1.243     0.562            0.554
45 m     1.252     0.683            0.322
```

Scale larghe alzano l'OR e appiattiscono le differenze fra squadre; sotto i
20 m il prior diventa quasi binario "in area o no". Il criterio va scritto
nel paper: senza, la scelta sembra arbitraria.

## D-05 · 11 ott 2026 · `geo_use_endpoint` = False

Prior su `min(partenza, arrivo)` invece della sola partenza.

```
                      singola corsa          squadra
corr con lead_to_shot  +0.403 -> +0.469
OR                      1.241 -> 1.255
split-half stress_per_run                +0.749 -> +0.628
```

Tre ipotesi alternative testate e scartate: endogeneità col containment
(`corr(geo_progress, containment) = -0.004`); doppio conteggio col terzo
fattore (corr prior-struttura +0.537 -> +0.532); riparametrizzazione della
scala (la correlazione fra i due prior satura a 0.843 per ogni scala tra 25
e 45 m).

Conclusione: **dove una corsa finisce è una proprietà dell'episodio, dove
parte è una proprietà di come la squadra si dispone, e solo la seconda si
ripete.** Il flag resta utile per animazioni e case study per-corsa.

Nota metodologica: la dissociazione **non** ha visto questo problema
(`targeted` e `received` reggevano in entrambe le configurazioni). Vedi D-10.

## D-06 · ott 2026 · `geo_ball_gate` = False

Attenuare il prior quando la palla è lontana dalla porta. Implementato,
spento: +0.001 di correlazione. Rumore.

Rischio aggiuntivo se riacceso: avvicina il prior alla fase di gioco, che
entra già indirettamente nella vulnerabilità strutturale — aumenterebbe la
ridondanza fra fattori.

## D-07 · ott 2026 · Containment = `separation_gain` z-scorato per subtype

Normalizzazione per subtype obbligatoria: il gain atteso è strutturalmente
diverso fra subtype (`cross_receiver` -1.25, `dropping_off` +0.52). Senza
normalizzare il modello premierebbe la difesa ogni volta che l'attaccante
corre verso un'area congestionata.

Limite noto e non risolto con XY: la separation **non distingue una scelta
tattica da un errore**. La diagnosi per subtype mostra dove cade — regge su
`coming_short` 1.80x, `behind` 1.50x, `support` 1.43x, `run_ahead` 1.37x;
crolla su `cross_receiver` 0.99x e `dropping_off` 0.57x.

## D-08 · 12 ott 2026 · Containment appreso = asse separato, non fattore

`containment = sigmoid((gain - previsione) / sd_residuo)`, con la previsione
di `separation_gain` da sola geometria pre-corsa, out-of-fold (GroupKFold su
`match_id`). Nessun esito entra: il target è `separation_gain`.

```
split-half containment_failure_mean   +0.026 -> +0.605
standardizzato per fase               -0.074 -> +0.519
confronto appaiato (stessi 400 split) +0.614 +/- 0.030, vince nel 98%
OR su lead_to_shot                     1.217 -> 1.229
corr con structural_vulnerability     -0.000  (nessun doppio conteggio)
dissociazione                          regge (targeted p=0.74, received p=0.41)
```

Perché funziona: lo z-score per subtype lasciava dentro il containment la
parte di `separation_gain` che la geometria spiega da sola (R² = 0.40) — e
quella parte dipende da *dove* è avvenuta la corsa, quindi varia da partita a
partita: per la stima di squadra è rumore.

**Ma dentro il prodotto dello stress la metrica standardizzata per fase
peggiora** (-0.110 ± 0.008, vince nel 9%). Quindi: `CONTAINMENT_KIND`
resta `separation_gain` di default, e il containment appreso vive nel
profilo a due assi (`team_two_axis.csv`: `esposizione` / `cedimento`).

Questa è la decisione architetturale più importante presa finora: il DRS non
è solo un prodotto a tre fattori, ha un secondo asse.

## D-09 · ott 2026 · Curve containment off

Containment dalla forma di `separation(t)` (`escape = sep_late - sep_min`)
invece dal solo `gain = end - start`. Su un match: AIC 301.0 contro 297.6 del
baseline — sei parametri in più per niente, correlazioni parziali al netto di
fase e subtype tutte sotto 0.08.

Cambierebbe idea: una rivalutazione sui 20 match (mai fatta). Il codice c'è.

## D-10 · 11 ott 2026 · Dissociazione e split-half servono entrambe

Coprono fallimenti **diversi**:

| test | cosa esclude |
|---|---|
| dissociazione | che il DRS misuri la *scelta dell'attaccante* |
| split-half | che il DRS misuri l'*episodio* invece di un tratto di squadra |

D-05 è la prova: `geo_use_endpoint` migliorava la dissociazione e peggiorava
la split-half, e la dissociazione da sola non l'avrebbe mai visto.

Nel paper va dichiarato: un fattore che alza l'OR e abbassa la split-half ha
migliorato la descrizione del singolo episodio e peggiorato il profilo di
squadra. Il paper si gioca sul secondo.

## D-11 · 18 ott 2026 · Igiene del repo e run di riferimento ricostruito

Tre interventi, nessuno sul modello.

**`.gitattributes` con `* text=auto`** (commit `c43fadd`). Il working tree su
Windows aveva CRLF e l'indice LF: `git status` segnalava 129 file modificati,
diff di 27065 inserimenti e 27065 cancellazioni interamente whitespace
(`git diff -w` vuoto). Qualsiasi modifica reale era invisibile. Nessun
`--renormalize` necessario, l'indice era già corretto: zero file riscritti.

**Tracking XY recuperato dagli oggetti LFS.** I 20 `*_tracking_extrapolated.jsonl`
sotto `data/matches/` sono puntatori git-lfs da 136 byte, e `.gitattributes`
non è mai esistito nella storia del repo, quindi il filtro LFS non è mai
stato attivo: i dati veri (1.7 GB) non erano su disco. Il tracking root che
aveva prodotto il run originale era `E:/Projects/opendata`, che non esiste
più. Gli oggetti LFS sono però ancora scaricabili dal fork: ora stanno in
`data/tracking_raw/` (gitignored, rigenerabile).

**`out/ref` rigenerato** con la config di default e `--oos`, copertura
tracking 100% su tutti i 20 match (zero corse con curva non calcolabile). La
dissociazione riproduce i numeri documentati alla terza cifra.

Tre cose già implementate e mai dichiarate, emerse rigenerando: la tabella di
stabilità contiene già la **correzione di Spearman-Brown**, già riporta i
**baseline** (`n_runs`, `prior_mean`, `deep_phase_share`), e il **profilo a
due assi** esiste in `team_two_axis.csv`.

## D-12 · 18 ott 2026 · Anisotropia angolare sulla partenza — scartata

Sostituire la distanza dalla porta con l'ampiezza angolare della porta vista
dal punto di partenza (pali a y = ±3.66), come fa Umemoto et al. 2022 sul
termine `S_r` di OBSO.

```
variante                  corr     Q1     Q4  sh prior
base  exp(-d/25)       +0.2282  0.044  0.231    +0.701   <- vince
d x angolo^0.5         +0.2058  0.043  0.224    +0.698
angolo  theta          +0.1868  0.051  0.218    +0.690
d x angolo             +0.1777  0.045  0.224    +0.672
d x angolo^2           +0.1199  0.046  0.224    +0.535
```

Degrada in modo monotono col peso dato all'angolo, sia sulla singola corsa
sia sulla stabilità per squadra. Per famiglia peggiora **sempre**:

```
famiglia             n  shot%     base    x ang   delta
costruzione       2095  0.024  +0.2824  +0.1939 -0.0885
finalizzazione    1517  0.249  +0.0812  +0.0663 -0.0150
laterali          1312  0.091  +0.1015  +0.0842 -0.0173
progressione      3953  0.133  +0.2360  +0.2228 -0.0132
in area            265           -0.0534 -0.0461 +0.0073
fuori area        8612           +0.2075 +0.1766 -0.0309
```

**Perché, e perché va nel paper.** Il termine angolare viene dagli xG, dove
la domanda è "quanto è buono un tiro da questo punto". Il target qui è
`lead_to_shot`: la corsa *contribuisce* a un tiro che avviene altrove, quindi
l'ampiezza della porta vista dalla partenza risponde alla domanda sbagliata.
Il danno è massimo sulla costruzione, dove l'angolo è piccolo e schiaccia
verso zero informazione che `exp(-d/25)` conservava; e le laterali perdono
perché sono pericolose in quanto generano cross, non tiri da dove partono.
Importare la geometria degli xG in un modello di corse senza palla è un
errore di categoria, e qui è misurato.

## D-13 · 18 ott 2026 · Angolo al punto di arrivo — scartato, ma informativo

L'angolo alla **fine** della corsa, non all'inizio. Ipotesi: se l'angolo
iniziale sbaglia perché la corsa non tira da lì, alla fine è il punto giusto.

L'ipotesi è corretta sulla singola corsa — l'angolo finale batte sia quello
iniziale sia il baseline:

```
variante                        corr     Q1     Q4  sh prior
exp(-dmin/25)  [D-05]        +0.3425  0.014  0.301    +0.599
angolo FINE                  +0.2733  0.027  0.290    +0.550
angolo MAX(inizio,fine)      +0.2687  0.026  0.283    +0.576
exp(-de/25) x ang.FINE       +0.2583  0.011  0.299    +0.556
base exp(-ds/25)             +0.2287  0.044  0.231    +0.692
angolo INIZIO                +0.1880  0.051  0.219    +0.671
```

Scartato per due motivi indipendenti.

**Stesso muro di D-05.** La split-half crolla da +0.692 a +0.550:
l'informazione sul punto di arrivo è una proprietà dell'episodio, non della
squadra. Identico al pattern di `geo_use_endpoint`.

**È ridondante con una strada già chiusa.** `corr(angolo FINE,
exp(-dmin/25)) = +0.863`, e la distanza minima da sola fa meglio dell'angolo
finale (+0.343 vs +0.273). Combinarli degrada (+0.257). L'angolo finale non
aggiunge niente oltre a quello che `geo_use_endpoint` già catturava, e
`geo_use_endpoint` è spento per la ragione di D-05.

Con questo il prior geometrico è **al suo tetto con la sola posizione di
partenza**: tutte e tre le vie di raffinamento della formula sono chiuse
(D-06 gate della palla, D-12 angolo iniziale, D-13 angolo finale), e la
scala è calibrata con criterio (D-04). Il margine residuo non è nella
formula, è in *dove* la si applica (Q-02) o in un prior esterno (Q-03).

## D-14 · 18 ott 2026 · Fattore T_r (distanza palla-corridore) — adottato in prova

Quarto fattore moltiplicativo nel prior: `T = exp(-ball_dist_runner_end / 20)`.
È il termine **T_r di OBSO** (Spearman 2018: probabilità che il prossimo
evento avvenga lì, gaussiana centrata sulla palla, σ ≈ 14 m) — l'unico dei
tre fattori di OBSO che il DRS non aveva.

```
                            baseline   s=10   s=20   s=30   s=45   s=70
OR lead_to_shot                1.217  1.200  1.215  1.218  1.219  1.218
split-half stress_per_run     +0.724 +0.759 +0.768 +0.758 +0.749 +0.745
split-half PHASE-ADJ          +0.534 +0.610 +0.672 +0.653 +0.627 +0.603
  Spearman-Brown              +0.696 +0.758 +0.804 +0.790 +0.771 +0.752
prior_mean                    +0.721 +0.750 +0.766     —      —      —
dissociazione targeted (p)      0.76   0.44   0.58      —      —      —
dissociazione received  (p)     0.44   0.94   0.76      —      —      —
```

**Primo intervento che non paga il prezzo di D-05.** OR invariato (1.215 vs
1.217, IC sovrapposti) e split-half **su**, con il salto maggiore sulla
metrica standardizzata per fase: +0.534 -> +0.672, Sp.-Brown +0.696 ->
+0.804. È la metrica di Q-01, l'unica che `deep_phase_share` non può
disputare. La dissociazione regge: `targeted` e `received` restano nulli,
quindi il timore che T_r importasse la scelta dell'attaccante è infondato.

**Non è smoothing:** c'è un massimo interno a 20 m. Verso 70 m il fattore
tende a 1 e il modello torna al baseline, come deve. Il picco cade nello
stesso ordine di grandezza del σ ≈ 14 m di Spearman, trovato
indipendentemente dai dati.

**Perché questa quantità e non le altre.** `ball_dist_runner_end` è una
distanza **relativa** fra due giocatori; `ball_dist_goal_end` è **assoluta**
rispetto al campo. Screening su 44 feature numeriche con tre criteri
(correlazione con l'esito, correlazione parziale al netto di fase × subtype,
split-half della media per squadra, più `r_depth` = correlazione della media
di squadra con `deep_phase_share`):

```
feature                  corr_y  pcorr_y      sh  r_depth
ball_dist_goal_end       -0.355   -0.195  +0.553   -0.931   <- miglior predittore, PEGGIOR candidato
geo_dist_end             -0.321   -0.137  +0.577   -0.942
geo_dist_min             -0.315   -0.129  +0.602   -0.931
geo_x_end                +0.307   +0.116  +0.561   +0.950
ball_dist_goal_start     -0.276   -0.100  +0.623   -0.871
ball_dist_runner_end     -0.093   -0.077  +0.660   -0.604   <- unico nel quadrante buono
```

Le distanze **assolute** hanno `r_depth` fra -0.87 e -0.95: la loro media per
squadra *è* la quota di fasi profonde. Alzano l'OR e affondano la split-half
perché sono profondità sotto altro nome, e la profondità non è un tratto che
il DRS possa rivendicare (Q-01). Unico candidato su 44 a passare i tre
filtri insieme: `ball_dist_runner_end`.

**Da confermare prima di fissare il default:** 8 squadre con >=3 match e IQR
sovrapposti fra le scale, quindi il picco a 20 m è plausibile ma non risolto.
Serve un confronto appaiato sugli stessi 400 split (come fatto in D-08, che
dava ±0.030) e la verifica che `ball_dist_runner_end` sia definita su tutte
le 8877 corse, per non reintrodurre popolazioni miste come `geo_dist_min`
in D-05.

## D-15 · 18 ott 2026 · Tassonomia dell'anisotropia: tre famiglie, tre esiti

Screening delle fonti di anisotropia mai testate, con gli stessi criteri di
D-14. `p/c` = |pcorr/corr|, quanto della correlazione sopravvive al controllo
per fase × subtype — cioè quanta informazione è davvero nuova.

```
candidato                       cop.   corr_y  pcorr_y   p/c      sh  r_depth
ang. al vertice PALLA           100%   +0.115   +0.112  0.98   +0.141   -0.216
ang. al vertice CORRIDORE       100%   -0.170   -0.116  0.68   +0.222   +0.105
T_r = exp(-br/20)  [D-14]       100%   +0.097   +0.080  0.83   +0.685   +0.542
heading verso porta (cos)       100%   +0.215   +0.065  0.30   +0.486   +0.631
|y_end| (laterale assoluto)     100%   -0.145   -0.076  0.52   +0.301   -0.133
ang. al vertice PORTA           100%   +0.156   +0.011  0.07   +0.299   +0.310
|y_start| (laterale assoluto)   100%   -0.032   -0.022  0.68   -0.084   +0.005
angle_of_engagement               0%   colonna vuota nel merged
goal_side_start / _end            0%   colonna vuota nel merged
```

Gli angoli del triangolo palla / corridore / porta sono calcolati dai soli
tre lati (`ball_dist_goal_end`, `ball_dist_runner_end`, `geo_dist_end`) con
il teorema del coseno: non servono le coordinate della palla, che il dataset
non porta.

**Tre famiglie con comportamenti opposti.**

| famiglia | esempio | informazione nuova | profondità | stabile |
|---|---|---|---|---|
| posizionale **assoluta** | distanza/angolo verso la porta | media | **sì** (0.87–0.95) | no |
| relativa **direzionale** | angoli del triangolo | **quasi tutta** (p/c 0.68–0.98) | no (0.1–0.2) | **no** (0.14–0.22) |
| relativa **scalare** | `ball_dist_runner_end` | buona (p/c 0.83) | media (0.54) | **sì** (0.69) |

L'angolo al vertice **palla** ha `p/c = 0.98`, il valore più alto di tutto
quello che è stato testato: la sua correlazione con l'esito è informazione
interamente nuova, e con `r_depth = -0.216` non è profondità travestita. Ma
la split-half è +0.141: pura informazione d'episodio.

**Conseguenza sull'architettura.** Non esiste un singolo fattore che alzi
l'OR *e* la split-half: c'è una frontiera, e le tre famiglie stanno in punti
diversi. Le direzionali relative sono lo strumento giusto per la
**dissociazione**, i case study e le animazioni; le scalari relative per
l'**aggregazione di squadra**. È la stessa conclusione di D-08 raggiunta da
un'altra strada: il modello ha due lavori diversi e un fattore solo non può
farli entrambi.

**Nota sui dati:** `angle_of_engagement`, `goal_side_start` e `goal_side_end`
sono vuote al 100% nel dataset unito — coerente con le 11 colonne
dell'enriched popolate solo su altri `event_type`. La goal-side era stata
testata su un singolo match in una sessione precedente; per riusarla va
ricalcolata dal tracking XY.

## D-16 · 18 ott 2026 · Direzionali ancorate alla partenza — scartate (Q-06 chiusa)

Estratte dal tracking XY le velocita' di corridore e difensore al frame di
**partenza** della corsa (`extract_velocity.py` -> `data/velocity_features.csv`,
8877 righe, copertura 99.8%; 96.7% sui coseni del difensore per il gate di
velocita').

Differenze finite a 10 Hz, finestra **backward** di 0.5 s
(`v = (p(t0) - p(t0-5)) / 0.5`): usa solo frame precedenti all'inizio della
corsa, quindi strettamente pre-evento come il DRS(t) misurato nei 3 s prima.
Variante centrale (±0.2 s) come controllo di sensibilita': stessi risultati.
I coseni sono invarianti all'orientamento del campo, quindi si lavora in
coordinate grezze definendo la porta attaccata per corsa da `attacking_side`.

```
candidato                   cop.   corr_y  pcorr_y   p/c      sh  r_depth
vel_run_cos_goal            100%   +0.202   +0.045  0.22  +0.524   +0.570
vel_run_cos_goal_centr      100%   +0.205   +0.049  0.24  +0.514   +0.584
vel_runner_speed            100%   +0.070   +0.047  0.68  +0.554   +0.298
vel_def_speed               100%   +0.034   +0.027  0.80  +0.634   -0.450
vel_rel_speed               100%   +0.006   -0.000  0.05  +0.548   +0.680
vel_closing                 100%   -0.018   +0.011  0.59  +0.019   -0.125
vel_def_cos_runner           97%   -0.079   -0.011  0.13  +0.607   -0.704
vel_def_cos_owngoal          97%   -0.159   -0.007  0.04  +0.484   -0.694

quadrante buono (|pcorr|>=0.05, sh>=0.60, |r_depth|<=0.70):  NESSUNO
```

**Due modi di fallire diversi, ed entrambi informativi.**

*La direzione del corridore è ridondante col subtype.* `vel_run_cos_goal` ha
una correlazione grezza rispettabile (+0.202) ma `p/c = 0.22`: quasi tutto
spiegato da fase × subtype. La causa, misurata:

```
quanta varianza del candidato e' gia' spiegata da...
candidato                  subtype     fase  sub x fase
vel_run_cos_goal             0.702    0.236       0.722
vel_def_cos_runner           0.096    0.032       0.108
vel_runner_speed             0.015    0.013       0.043
```

**Il 70% della direzione di corsa è già nel subtype.** Ha senso: i 10
subtype di SkillCorner *sono* etichette direzionali (`behind`, `run_ahead`,
`dropping_off`, `coming_short`). Una feature di velocita' direzionale e' la
versione continua di una categorica che la logistica di dissociazione
controlla **obbligatoriamente** (D-07). Non puo' aggiungere niente per
costruzione.

*Il proxy 2D del cono visivo non predice.* `vel_def_cos_runner` — quanto il
corridore e' fuori asse rispetto alla direzione di moto del difensore — e'
genuinamente nuovo (solo 9.6% spiegato dal subtype) e con `sh = +0.607` e'
anche ragionevolmente stabile. Ma `pcorr = -0.011`: non predice l'esito.

Conseguenza pesante sul progetto: **la direzione di moto del difensore non
e' un sostituto utilizzabile dell'orientamento del corpo.** Era l'alternativa
degradata all'awareness di Filippo da usare su tutti i 20 match; non
funziona. Resta solo il body pose, disponibile su 2 match, con la split-half
per squadra impossibile e la via alternativa a livello di GIOCATORE (~800
corse) come unica validazione.

**Q-06 è chiusa negativamente, e la frontiera di D-15 regge.** L'ipotesi era
che ancorare una direzionale alla partenza la rendesse informativa *e*
stabile. Non la rende ne' l'una ne' l'altra: ancorata all'inizio la direzione
e' già nel subtype, ancorata alla fine è nuova ma legata all'episodio. La
tassonomia completa:

| quantita' | nuova? | stabile? | perche' |
|---|---|---|---|
| direzionale alla **partenza** | no (p/c 0.22) | media | già nel subtype (R² 0.70) |
| direzionale alla **fine** | sì (p/c 0.98) | no | proprietà dell'episodio |
| scalare **relativa** | sì (p/c 0.83) | sì | unica usabile → D-14 |
| posizionale **assoluta** | media | no | è profondità (r_depth 0.93) |

`T_r` resta l'unico guadagno trovato. Il prior è al suo tetto con i dati XY.

## D-17 · 18 ott 2026 · Il profilo a due assi va calcolato col containment appreso

Il `team_two_axis.csv` di `out/ref` usa `containment_kind="separation_gain"`.
Con quella config il secondo asse non e' utilizzabile come figura:

```
                            sep_gain   learned
cedimento: range             0.034      0.044
cedimento: CV                0.020      0.025
corr esposizione-cedimento  +0.659     +0.427
corr cedimento-deep_share   +0.413     +0.221
containment_phase_adj       -0.074     +0.517
stress_per_run_phase_adj    +0.534     +0.422
dissociazione OR             1.217      1.227   (nulli intatti)
```

Con `separation_gain` la figura sarebbe: un asse che e' altezza del blocco
(`corr esposizione-deep_phase_share = +0.880`), un secondo asse con
split-half standardizzata per fase **-0.074** — una monetina — e i due
correlati a +0.659. Una nuvola schiacciata su una diagonale.

Col containment appreso il secondo asse diventa riproducibile (+0.517) e in
buona parte ortogonale al primo (+0.427). Conferma D-08 da un'altra strada:
dentro il prodotto moltiplicativo il learned peggiora
(`stress_per_run_phase_adj` +0.534 -> +0.422), come **asse separato**
funziona. Le due configurazioni servono due scopi diversi e convivono:

- `out/ref` (separation_gain) -> lo stress e la dissociazione
- `out/ref_learned` -> il profilo di squadra e la figura

**Residuo da gestire in fase di figura:** `cedimento` resta compresso
(CV 0.025, tutte le squadre fra 0.476 e 0.520). Non e' degenerazione — e'
una media di una sigmoide su centinaia di corse, quindi centrata su 0.5 per
costruzione, ed e' riproducibile. Va presentato come z-score o rango
percentile con CI, non come valore grezzo, altrimenti la figura appare
piatta pur non essendolo.

## D-18 · 18 ott 2026 · Correzione: due discrepanze negli script di screening

Tracciando la provenienza della Figura 1 sono emerse due incoerenze **negli
script di analisi**, non nel repo. Nessuna conclusione cambia, ma i numeri
vanno annotati.

**1. Definizione di "fase profonda".** Gli script di screening di D-12, D-14,
D-15 e D-16 usavano `{low_block, defending_transition, defending_quick_break}`;
`cfg.deep_phases` nel repo e'
`{low_block, defending_set_play, defending_quick_break}`. Le due quote per
squadra correlano +0.905 (medie 0.437 vs 0.398). Effetto sulla colonna
`r_depth`:

```
feature                    r_depth (script)   r_depth (repo)   delta
ball_dist_goal_end                  -0.931           -0.936   -0.005
ball_dist_runner_end                -0.604           -0.551   +0.053
vel_def_cos_runner                  -0.704           -0.543   +0.160
vel_run_cos_goal                    +0.570           +0.272   -0.298
```

Le conclusioni reggono perche' nessuna poggiava su `r_depth`: D-14 su
`p/c = 0.83` e split-half +0.685, D-16 sul 70.2% di varianza spiegata dal
subtype e su `pcorr = -0.011`. Due soglie vanno pero' rilette: col
`deep_phases` del repo, `vel_def_cos_runner` (-0.543) e `vel_run_cos_goal`
(+0.272) **rientrerebbero** nel filtro `|r_depth| <= 0.70`, e restano
scartati solo per il criterio di informazione nuova. D-16 va letta cosi'.

**2. Aggregazione per squadra nella Figura 1.** Lo script della figura usava
la media **pesata sulle corse**; `aggregate.team_two_axis` usa la media
**semplice sulle partite**. Corr +0.9996, scarto massimo 0.0005 su
esposizione e 0.0011 su cedimento: visivamente nulla. Lo script e' stato
allineato alla definizione del repo (ogni partita pesa uguale, coerente con
Q-05: l'unita' di campionamento e' la partita).

Regola che ne discende: gli script di analisi importano le costanti da
`cfg`, non le ridichiarano.

## D-19 · 19 ott 2026 · Il containment resta: non predice la corsa, de-confonde la squadra

Test di controllo mai eseguito: rimuovere del tutto il containment
(`cf = 1` ovunque, quindi `stress = prior x struttura`) e rifare entrambe
le validazioni. L'ipotesi era che il fattore sottraesse invece di
aggiungere, vista la sua correlazione con l'esito di -0.016 e i quartili
piatti.

```
                            con cf    senza cf
OR lead_to_shot              1.217      1.217    invariato
  targeted (p)                0.76       0.76
  received (p)                0.44       0.44
split-half stress_per_run   +0.724     +0.680
split-half PHASE-ADJ        +0.534     +0.158    crolla
  Spearman-Brown            +0.696     +0.272
prior_mean                  +0.721     +0.721
```

**L'ipotesi era sbagliata, e il modo in cui lo e' spiega il fattore.** Il
containment non contribuisce **nulla** alla dissociazione: l'OR e' identico
alla quarta cifra con e senza. Ma togliendolo la split-half standardizzata
per fase passa da +0.534 a **+0.158**.

Senza containment lo stress e' `prior x struttura`, cioe' quasi solo
geometria: una volta standardizzata la fase non resta niente. **Il
containment e' l'unica componente dello stress che non sia geometria**, ed
e' quindi cio' che da' contenuto alla metrica di squadra al netto
dell'altezza del blocco (Q-01).

Due conseguenze.

Primo, il containment **resta nel prodotto**, e la motivazione cambia: non
sta li' per predire la singola corsa — non lo fa — ma per de-confondere la
metrica di squadra. Va detto cosi' nel paper, perche' un revisore che
guarda solo la sua correlazione grezza (-0.016) concluderebbe il contrario.

Secondo, e' l'argomento piu' forte a favore di D-10 (servono entrambi i
test): qui un fattore e' **invisibile** alla dissociazione e **decisivo**
per la split-half. Con la sola dissociazione lo avremmo tolto.

### Diagnostica per famiglia (proposta chiusa)

L'idea di separare il containment per famiglia nasceva dalla diagnosi che
la separation regge su alcuni subtype e crolla su altri. Correlazione
parziale col `lead_to_shot` al netto della fase, dentro ciascuna famiglia:

```
famiglia             n  sep_gain   learned
costruzione       2095    -0.026    -0.002
finalizzazione    1517    -0.030    +0.005
laterali          1312    -0.008    +0.038
progressione      3953    -0.027    +0.013
```

Nessuna famiglia mostra discriminazione utile, con nessuno dei due
containment: tutti i valori sono sotto 0.04 in modulo. Non c'e' una
famiglia "buona" da separare da una "cattiva" — **il containment non
discrimina la singola corsa in nessun regime**, coerente con il test di
sopra. Un `cf` per famiglia aggiungerebbe parametri senza segnale.
Proposta chiusa.

### Stato delle quattro proposte sul containment

| proposta | esito |
|---|---|
| awareness come disambiguatore | aperta, ma e' decisione di scope (D-16, Pose2Pitch) |
| proxy 2D del cono visivo | **scartata** — D-16, `pcorr = -0.011` |
| containment per famiglia | **scartata** — nessuna famiglia discrimina |
| rimuovere il containment | **scartata** — l'OR non cambia ma la split-half per fase crolla |

## D-20 · 19 ott 2026 · Anatomia della curva: tre numeri da correggere o dichiarare

Ricostruita e ispezionata la curva di un match (`DRS/figures/fig_curve.py`,
figura in `out/figures/curve_1996436.png`) con il supporto reso esplicito.
Tre misure diagnostiche (`diag_curve.py`, `diag_pre.py`).

### 1. La copertura dichiarata e' quella sbagliata — CORREZIONE

```
copertura media per MATCH    40.1%    <- quella che la pipeline stampa e il README cita
copertura media per SQUADRA  21.4%    <- quella che alimenta le metriche di squadra
```

Il 40.1% e' la copertura dell'asse temporale quando si considerano **tutte**
le corse del match. Ma `team_match_metrics` e `team_two_axis` costruiscono
una curva **per squadra difendente**, e quella copre il **21.4%** dei frame.

Nel paper va dichiarato il 21.4%, non il 40.1%: e' la copertura della curva
da cui vengono i numeri di squadra. Dichiarare il 40.1% sovrastima di circa
due volte e si verifica in trenta secondi.

### 2. La curva accumula davvero — dove esiste

```
frame supportati con >=2 corse attive:  56.6%
  sul totale dei frame del match:       12.1%
```

Piu' della meta' dei frame supportati ha almeno due corse che contribuiscono
insieme. La tesi dell'accumulo non e' decorativa: dove il DRS(t) e' definito,
nella maggioranza dei casi sta sommando episodi sovrapposti. Visivamente la
curva sembra una sequenza di impulsi isolati; il numero dice il contrario.

Ma vale solo sul 12.1% dei frame totali. Formulazione difendibile: **stress
per corsa che si accumula su corse sovrapposte e decade**, su un asse
temporale definito per un quinto della partita. Non "segnale difensivo
continuo".

### 3. Un terzo delle corse ha DRS(t) pre-corsa nullo — verificato, regge

```
corse con DRS(t) identicamente zero nei 3 s precedenti:  2850/8877 = 32.1%
shot rate   pre = 0  ->  0.080        pre > 0  ->  0.140
```

Il predittore della dissociazione e' **esattamente zero per un terzo del
campione**: un revisore puo' obiettare che l'OR misuri "c'era attivita'
recente oppure no" invece di un effetto graduato. Test:

```
                                            n      OR        IC 95%
tutte le corse (riferimento)             8877   1.226  [1.164, 1.292]
SOLO corse con pre > 0                   6027   1.242  [1.166, 1.323]
solo binaria pre>0 vs pre=0              8877   1.186  [1.101, 1.278]
```

**L'effetto e' graduato.** Dentro il sottoinsieme con stress precedente non
nullo l'OR e' 1.242, piu' alto del riferimento; la versione puramente
binaria e' piu' debole (1.186). Quindi il livello del DRS(t) porta
informazione oltre la semplice presenza. L'obiezione e' prevedibile e la
risposta va messa nel paper con questi tre numeri.

### Conseguenza sulle animazioni

Le GIF di `animate.py` sono una risorsa valida **a condizione che mostrino
il supporto**. Disegnare uno zero piatto dove la curva non e' definita
comunica "difesa sotto controllo" quando il dato dice "non misurato" — ed e'
il 78.6% del tempo di gioco per squadra. La figura statica usa una banda
grigia; l'animazione deve fare lo stesso, o limitarsi alle finestre
supportate.

## D-21 · 19 ott 2026 · Hero case identificato: match 1996436, min 29.4

La clip `drst_clip_v2_min29` mostra, senza dirlo, il caso che riassume
l'intera tesi del progetto.

```
  min event_subtype    famiglia       targeted received lead_to_shot  sep_gain
29.36 support          progressione      False    False         True     -0.50
29.36 behind           finalizzazione    False    False         True     +1.70
29.37 behind           finalizzazione    False    False         True     +1.51
```

**Tre corse, nessuna servita, e il tiro arriva lo stesso.** Picco DRS(t)
0.92, fase `low_block`, i due `behind` si liberano entrambi (`sep_gain`
+1.70 e +1.51). La difesa viene aperta da corse a cui nessuno passa la
palla, e il tiro nasce altrove.

E' **l'effetto decoy in un episodio** — il 68% dei `lead_to_shot` del
dataset — ed e' la dissociazione resa visibile: il DRS(t) sale, il tiro
arriva, e `targeted` e' falso su tutte e tre. Esattamente cio' che l'OR
nullo su `targeted` dice in forma aggregata.

E' l'hero case del paper e del video. Oggi la clip non lo dichiara: mostra
il picco e le corse, non il fatto che nessuna fosse l'obiettivo del
passaggio.

Per confronto, la clip `VID-20261013-WA0004` (match 1986691, min 4.9) ha
tre `cross_receiver` in low_block, una sola targeted, **zero lead_to_shot**:
e' stress senza conseguenza. Legittimo da mostrare come contro-esempio
("picco alto, nessun esito"), ma non e' una narrazione che si regge da sola.

### Correzioni alle animazioni, in ordine di valore

1. **Dichiarare l'esito.** Un'etichetta sulle corse (`non mirata`) e un
   marcatore `TIRO` al frame giusto. Senza, lo spettatore vede un picco e
   non sa perche' conta.
2. **Non disegnare il futuro.** Oggi la striscia mostra l'intera curva dal
   primo fotogramma, picco compreso: anticipa l'esito e rappresenta male
   cio' che la metrica sa in quell'istante. La curva va disegnata fino al
   cursore.
3. **Banda di supporto** (D-20): dove la curva non e' definita lo zero non
   significa "difesa sotto controllo".
4. **Leggibilita'**: a 1550x860 i nomi e i valori dei tre fattori sono
   illeggibili in un video. I numeri dei fattori sono il contenuto, vanno
   ingranditi; la colorbar "stress locale" si puo' togliere.
5. **Asse in mm:ss** invece dei minuti decimali (29.36).
6. **Nomi dei club** al posto di "in possesso / difende".

---

# Questioni aperte

## Q-01 · `deep_phase_share` batte il DRS sulla stabilità

La semplice quota di corse subite in fasi difensive profonde fa split-half
**+0.793** (Sp.-Brown +0.885), contro **+0.724** di `stress_per_run`. E
`prior_mean` da solo fa +0.721: il margine del DRS completo sul prior è
**+0.003**, su `n_runs` è +0.029.

Il prior è già noto essere spiegato al 52% dalla fase difensiva e al 78% dal
terzo di partenza, con correlazione 0.985 con la quota di corse in fasi
profonde.

La domanda: **esiste una componente del DRS stabile per squadra e
indipendente da dove la squadra difende?** Oggi la risposta nei numeri è
"quasi no". I metodi standardizzati per fase (OR 1.114, split-half +0.534)
sono il tentativo di isolarla.

Implicazione sul posizionamento del paper: se la risposta resta no, il
contributo non sta nella stabilità per squadra — sta nella **dissociazione**,
che il DRS vince (OR 1.217 su `lead_to_shot`, nullo su `targeted` e
`received`) e che un conteggio di fasi non può nemmeno disputare.

## Q-02 · Prior frame-level

Oggi `danger_prior` è **costante lungo tutta la corsa**, campionato a
`frame_start`; e `structural_vulnerability` ha lo stesso limite dichiarato
nella sua docstring ("attributo della corsa, non segnale di fondo continuo").

Un prior `p(t) = exp(-d(t)/25)` ricalcolato a ogni frame
(`indexing.DenseTrackingIndex`, già predisposto come stub) non è un
raffinamento della formula: cambia *dove* la si applica, e risolve il
conflitto partenza/arrivo invece di scegliere un lato — media e picco della
curva diventano due metriche distinte della stessa quantità, la prima
dominata dalla posizione di partenza (stabile), la seconda dall'episodio.

È anche il frame in cui lavorano OBSO e l'awareness di Filippo, che è
nativamente frame-level: oggi la si innesterebbe su un prior che non lo è.

Il tracking XY ora è su disco, quindi è tecnicamente fattibile.

## Q-03 · EPV SkillCorner come prior esterno

SkillCorner ha un EPV nel pacchetto Game Intelligence: probabilità di segnare
entro 30 secondi o prima dell'interruzione, da tracking, **con la fase di
gioco nello stato** (loro esempio: stessa zona del terzo medio, <2% in fase
Create e >20% in Transition).

Se è nel nostro accesso ai dati, dà in un colpo un prior validato
esternamente e non derivato dai nostri esiti, e il baseline di livello
industriale che oggi manca nella gerarchia dei confronti.

Da verificare: se l'EPV è incluso nei dati dell'Analytics Cup.

## Q-04 · Il package su disco è incompleto

8 dei 14 moduli non sono **mai** stati committati: `cli`, `priors`, `curve`,
`indexing`, `tracking_features`, `animate`, `plotting`, `evaluate_upgrade`.
`build_single_file.py` li elenca tutti e quattordici, quindi oggi rigenerare
il file unico fallisce.

Di conseguenza l'unica copia completa del modello è `DRS/DRS_t/drst.py` —
il file che il README dice di non modificare perché "è generato". Il
generatore non ha più i sorgenti.

`drst_v3.zip` nel progetto contiene il package completo, ma non è accertato
se sia più vecchio o più nuovo del file unico su disco (il commit si chiama
`drst_v2`, lo zip è `v3`, esiste anche un `v4`). Serve un confronto modulo
per modulo prima di sovrascrivere.

Collegato: `learned_containment.py` esiste in due copie identiche (`drst/` e
`DRS/DRS_t/`, stesso md5) e non è generato dal build, quindi ogni fix va
applicato due volte a mano.

## Q-05 · Rivalutare i CI per il clustering per partita

Gli intervalli di Wilson calcolati su 8877 corse trattandole come
indipendenti (tabelle per subtype, decoy 8.5% [7.95, 9.12]) sono **troppo
stretti**: l'unità di campionamento reale è la partita, non la corsa.

Su una metrica di squadra la deviazione standard d'errore stimata per cluster
di partita è risultata **2.33x** quella binomiale. Nessuna conclusione
qualitativa cambia (gli effetti grossi restano grossi), ma è una correzione
che un revisore attento nota. Rimedio: bootstrap a blocchi su `match_id`,
che è già la struttura usata nella split-half.

## Q-06 · CHIUSA (D-16) · direzionale ancorata alla partenza

Testata e scartata: vedi D-16. La direzione del corridore alla partenza e'
spiegata al 70% dal subtype; il proxy 2D del cono visivo del difensore e'
nuovo ma non predice.

## Q-07 · Pitch control come anisotropia, e cosa resta

Con D-12, D-13 e D-16 tutte le fonti di anisotropia calcolabili dai dati
attuali sono esaurite. Quello che resta e' di un ordine diverso:

**Pitch control** (Spearman), cioe' il termine `C_r` di OBSO: lo spazio
effettivamente controllabile davanti al corridore, dai tempi di arrivo di
tutti i 22. E' l'unico modo che la letteratura considera definitivo per
rompere l'isotropia, ed e' per costruzione relativo e non posizionale —
quindi l'unica famiglia che in D-15 e' risultata utilizzabile. Costo alto,
e con 8 squadre da validare il rapporto rischio/ricompensa va pesato contro
la deadline del 18 dicembre.

**Awareness da body pose** (Filippo). D-16 ha chiuso l'alternativa
degradata, quindi questa non ha piu' un fallback: o il body pose o niente.
Disponibile su 2 match, da validare a livello di giocatore su ~800 corse.

**EPV SkillCorner** (Q-03), che incorpora la fase nello stato e sarebbe un
prior esterno validato.
