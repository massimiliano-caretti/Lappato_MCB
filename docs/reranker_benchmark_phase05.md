# Phase 0.5 — micro-benchmark del reranker stdlib

> **Scopo del documento.** Riportare onestamente cosa una versione minimale del reranker (senza modello di embedding, senza dipendenze nuove) cambia rispetto al baseline lessicale del core, sui 35 domini bundlati. Il risultato decide se Phase 1/2 (extras `[embed]` con MiniLM via ONNX) è giustificato o se basta un reranker lessicale fine-grained nel core.

## Setup, in una pagina

| Voce | Valore |
|---|---|
| Reranker | [`lappato_mcb.rerankers.trigram.TrigramJaccardReranker`](../lappato_mcb/rerankers/trigram.py) — Jaccard su trigrammi di carattere |
| Dipendenze nuove | nessuna (riusa `fingerprint.py`) |
| Domini | 35 (tutti i gold-set bundlati) |
| Topic totali | 505 |
| Paper sintetici per topic | 20 (1 rilevante + 19 rumore) |
| Random seed | 42 (run completamente deterministico) |
| Script | [`examples/benchmark_reranker_stdlib.py`](../examples/benchmark_reranker_stdlib.py) |
| Test di regressione | [`tests/test_trigram_reranker.py`](../tests/test_trigram_reranker.py) |
| Run completo | ~1 secondo, CPU singolo core |

### Due scenari

**Easy (controllo).** Per ogni topic del gold-set, sintetizziamo un paper rilevante che contiene **verbatim** le `must_match_keywords` + 19 paper di rumore con titoli random. Il baseline lessicale dovrebbe già fare 1.000.

**Hard (caso di interesse).** Per ogni topic, sintetizziamo un paper rilevante che usa **varianti morfologiche** delle keyword (`calibration` → `pre-calibrationation`, `framewise` → `sub-framewisation`) — il keyword originale **non è presente nel titolo**. I 19 paper di rumore hanno il 30% di probabilità di contenere uno dei keyword originali, in modo da **batterci il baseline lessicale**: il rumore ha overlap=1, il rilevante overlap=0.

### Cinque modi di blending testati

| Modo | Formula | Ruolo |
|---|---|---|
| `lex` | solo punteggio token-overlap (= core attuale senza reranker) | baseline |
| `additive` | `lex + w · max(rerank − floor, 0)` con `w=0.5, floor=0.30` | **proposto in Phase 0** |
| `rrf` | RRF balanced (Cormack et al. 2009): `1/(60+rank_lex) + 1/(60+rank_rerank)` | alternativa standard del settore |
| `rrf_2x` | RRF con peso `2·` sul reranker | "semantic-first hybrid" |
| `rerank_only` | solo punteggio reranker (non-blend) | upper bound teorico |

---

## Risultati (numeri precisi, run deterministico)

### Scenario EASY — controllo di sicurezza

Tutti i modi raggiungono Recall@5 = Recall@10 = **1.000**.

Conclusione: nessun blend degrada lo scenario facile. Il reranker non genera regressioni.

### Scenario HARD — il caso che importa

| Modo | R@5 | R@10 | Δ R@5 vs lex | Δ R@10 vs lex |
|---|---:|---:|---:|---:|
| `lex` (baseline) | 0.4897 | 0.9866 | — | — |
| `additive` (Phase 0 default) | 0.4897 | 0.9866 | **+0.0000** | **+0.0000** |
| `rrf` (balanced) | 0.7557 | 0.9941 | **+0.2660** | +0.0076 |
| `rrf_2x` (lex + 2× rerank) | **0.8298** | 0.9962 | **+0.3401** | +0.0096 |
| `rerank_only` (upper bound) | 0.8872 | 0.9959 | **+0.3975** | +0.0094 |

### Cosa significano questi numeri

Tre fatti distinti:

1. **Il reranker da solo ha segnale fortissimo**: passa da 0.49 a 0.89 su R@5. Trecentonovantacinque centesimi di recall recuperati su 505 topic. Il segnale c'è.

2. **Il blend additivo del Phase 0 lo schiaccia completamente**: delta = +0.0000. Il motivo lo abbiamo verificato con uno spot check (vedi `Diagnostica`): il punteggio lessicale grezzo (token-overlap intero, valori 0/1/2/...) sovrasta sempre il delta di Jaccard del reranker (valori frazionari 0.05-0.20). Il `floor` di 0.30 castra ulteriormente il poco segnale residuo. **Il blend additivo è strutturalmente inadatto a fondere segnali con scale così diverse.**

3. **RRF (rank fusion) sblocca il segnale del reranker**: balanced lifta R@5 di +0.27, e con peso doppio sul reranker arriva a +0.34. RRF è scale-invariante per costruzione (lavora sui rank, non sui punteggi), quindi non è ingannato dalle scale eterogenee.

### A K=10 il gap si comprime

Il delta sul Recall@10 è circa 10× più piccolo del delta sul Recall@5 (+0.01 vs +0.34). Coerente con l'intuizione: a K=10 il rumore ha già "scaricato" abbastanza candidati che il rilevante entra comunque nei primi 10. **Il valore del reranker è soprattutto sui top results visibili all'utente** (top-5 del weakness card).

---

## Diagnostica: perché il blend additivo non funziona

Spot-check sul primo topic di `neuroscience` (`framewise_displacement`, query = `"framewise displacement"`):

| relevant | lex | rerank | titolo (troncato) |
|---|---:|---:|---|
|   | 1.0 | 0.157 | Survey of for the in proposed theoretical **displacement** |
|   | 1.0 | 0.133 | The overview novel comparison **framewise** |
|   | 1.0 | 0.129 | Novel system proposed unrelated **framewise** |
| **★** | 0.0 | **0.205** | Hard paper #1: sub-framewisation displacemenising |
|   | 0.0 | 0.051 | Evaluation novel overview framework approach |

Il reranker **assegna correttamente il punteggio più alto** al paper rilevante (0.205 vs 0.157 dei decoy). Il blend additivo `lex + w·max(rerank−floor, 0)` non lo sfrutta:

- per il rilevante: `0 + 0.5·max(0.205 − 0.30, 0) = 0`
- per i decoy: `1 + 0.5·max(0.157 − 0.30, 0) = 1`

I decoy lessicali vincono a parità di reranker per via del floor che castra il segnale del rilevante.

Anche **rimuovendo il floor**: il rilevante ottiene `0 + 0.5·0.205 = 0.10`, i decoy ottengono `1 + 0.5·0.157 = 1.08`. Il delta del reranker è **un ordine di grandezza** sotto il lex score. Il blend additivo non può ribaltare.

RRF lavora sul rank (1-based) di ciascun ranker e non subisce questo problema:
- rilevante: rank 1 nel reranker → contribuisce `1/61 ≈ 0.0164`; rank 4-5 nel lex → contribuisce `1/64 ≈ 0.0156`
- decoy "displacement": rank 2 nel reranker → `0.0161`; rank 1 nel lex → `0.0164`
- la somma sposta il rilevante in cima.

---

## Implicazioni per il design del core

Phase 0 ha aggiunto al core un **`Reranker` Protocol** + un **blend additivo**. Phase 0.5 dimostra che:

| Componente | Verdetto Phase 0.5 |
|---|---|
| `Reranker` Protocol | ✅ utile e ben fatto. Permette plug-in senza toccare il core. |
| Constructor arg `reranker=...` | ✅ utile, manteniamo. |
| Audit trail nei weakness card | ✅ utile, manteniamo. |
| Graceful degrade su crash / NaN / wrong length | ✅ corretto, manteniamo. |
| **Blend additivo `lex + w · max(rerank − floor, 0)`** | ⚠️ **strutturalmente inadeguato** — non sfrutta il segnale di un reranker che lavora su scala diversa. |

**Raccomandazione del team per Phase 0.6** (modifica al core, ancora zero deps nuove):

Aggiungere un **`blend_mode` argument** al costruttore:

```python
LAPPATO_MCB(
    project_root=...,
    manifest=...,
    run_tag=...,
    reranker=...,                 # già presente in Phase 0
    blend_mode="additive",        # default invariato (back-compat)
                                  # alternativa: "rrf" con peso 2x sul reranker
)
```

Manteniamo l'`additive` come **default** per backward compatibility con utenti che hanno già configurato i propri reranker in Phase 0; ma documentiamo nel README che **per usi reali con un reranker semantico si raccomanda `blend_mode="rrf"`**. La scelta di default conservativa rispetta il principio "non snaturare LAPPATO_MCB"; chi vuole il segnale lo abilita esplicitamente.

---

## Implicazioni per Phase 1/2 (extras `[embed]`)

Tre considerazioni in tensione:

### Argomento PRO lo shipping di `[embed]` (MiniLM ONNX)

- Il trigram-reranker, pur essendo lessicale fine-grained, dimostra che **un reranker che lavora su rappresentazione diversa dal token-overlap recupera ~30 punti R@5 sui casi hard**.
- Un modello di embedding semantico (MiniLM) catturerebbe **anche i sinonimi veri** (`vehicle`/`car`, `error`/`mistake`), non solo le varianti morfologiche. Aspetterò un uplift **almeno pari** a quello del trigram, plausibilmente maggiore.
- Il design di plug-in del core (Phase 0) resta intatto: `[embed]` ship un `OnnxMiniLMReranker` che è un drop-in del Protocol, esattamente come `TrigramJaccardReranker` lo è oggi.

### Argomento CONTRO

- Il trigram-reranker **stdlib-only** già fornisce un reranker decente per i casi di vocabolario morfologicamente correlato — il caso più frequente dei manifest scientifici (`calibration`/`calibrating`, `regression`/`regressed`).
- Aggiungere 120 MB di dipendenze (`onnxruntime` + `tokenizers` + modello ONNX) per coprire i sinonimi puri **non testati**: non sappiamo quanto frequenti siano nei manifest reali. Potrebbe essere un caso raro.
- Il `TrigramJaccardReranker` è zero-deps, già shipped, e con `blend_mode="rrf"` recupera già il +0.34 sui hard cases.

### Decisione raccomandata dal team

**Procedere a Phase 0.6** (aggiungere `blend_mode="rrf"` al core, ancora zero deps), poi **fermarsi e raccogliere segnale dagli utenti reali** prima di Phase 1/2.

Senza dati su quanto frequenti siano i casi sinonimo-puro nei manifest reali, lo sforzo di Phase 1/2 (export ONNX, packaging extras, NOTICE files, ~120 MB) potrebbe non essere giustificato. Il `TrigramJaccardReranker` + RRF copre la maggior parte del beneficio empiricamente misurabile su questo benchmark.

**Phase 1/2 si riapre se** un utente reale segnala fallimenti di retrieval che il trigram non risolve, o se introdurremo manifest in domini con vocabolario semantico più variegato (es. legal, medico-clinico).

---

## Limiti onesti di questo benchmark

| Limite | Conseguenza |
|---|---|
| Corpus 100% sintetico, generato dai gold-set stessi | I numeri sono un floor di self-consistency, non una misura di retrieval reale |
| Le "varianti morfologiche" sono prefissi/suffissi standard | Il reranker semantico (MiniLM) qui è probabilmente *sotto*-stimato perché non testiamo sinonimi puri |
| Il rumore è generato da un dizionario di 25 parole funzionali | I decoy lessicali non rappresentano la varietà del rumore arXiv reale |
| 20 paper per topic, fissi | Diversi K e diverse densità di rumore potrebbero invertire qualche risultato |
| Il segnale lessicale è token-overlap intero (=BM25-lite) | Il lex `_score_hit` reale del core è più ricco (recency, citazioni); i numeri qui sono uno stress-test del "solo lessicale" |
| Risultati su synthetic ≠ risultati online | Sui veri arXiv harvests, il lessicale è probabilmente più forte (la query ha più token discriminanti); il delta atteso del reranker resta plausibilmente positivo, magnitude ignota |

I test di regressione in `tests/test_trigram_reranker.py` pinzano i numeri così come misurati qui: se driftano, sapremo che qualcosa è cambiato.

---

## Conclusione

Phase 0.5 ha dimostrato **tre cose, in ordine di importanza**:

1. **Il segnale del reranker esiste**, ed è grosso (+0.40 R@5 su 505 topic hard). Il dubbio "vale la pena un reranker?" è risolto: sì.
2. **Il blend additivo del Phase 0 lo spreca**: delta +0.0000. Strutturalmente inadeguato.
3. **Il fix è banale**: cambiare il blend da additivo a RRF (scale-invariante) recupera ~80% del segnale (+0.34 vs +0.40 dell'upper bound). RRF è zero-deps, sta in ~30 righe nel core, e non degrada lo scenario easy.

**Prossimo passo proposto**: implementare Phase 0.6 — `blend_mode="rrf"` come opzione nel core, con `additive` ancora come default per backward compatibility. **Dopo** Phase 0.6, decidere su Phase 1/2 (extras `[embed]`) sulla base di feedback degli utenti reali, non in anticipo.

Tabella riassuntiva delle proposte rivedute alla luce di Phase 0.5:

| Phase | Cosa | Costo | Beneficio dimostrato |
|---|---|---|---|
| Phase 0 (fatta) | Reranker Protocol nel core | 0 deps, 30 LOC | abilita plug-in |
| Phase 0.5 (fatta) | Trigram reranker + benchmark | 0 deps, ~200 LOC | identifica il problema del blend |
| **Phase 0.6** (proposta) | `blend_mode="rrf"` nel core | 0 deps, ~30 LOC | **+0.34 R@5 dimostrato** |
| Phase 1 (proposta originale) | Export ONNX MiniLM | runtime di build | non dimostrato finché non c'è feedback utente |
| Phase 2 (proposta originale) | Extras `[embed]` con MiniLM | ~120 MB deps | non dimostrato |
| Phase 4 (gate decisionale) | Benchmark online reale | rete + quota API | l'unico modo di chiudere il dubbio |

La proposta del team per la prossima iterazione è **Phase 0.6**, non Phase 1.
