# 2026-09-24 — Bugfixes und Optimierungen

Aus einer Durchsicht der Hot-Path-Module (`qwen3.rs`, `qwen3_5.rs`,
`weights.rs`, `backend.rs`). Die Gewinne unten sind **geschätzt, nicht
gemessen** — jede Optimierung gilt erst, wenn `scripts/sweep.sh` bzw.
`examples/measure` sie auf einem echten Checkpoint zeigt und die Antworten
gleich bleiben.

## Fortschritt

- ✅ Task 1: `with_chunk_size` wirkt nicht (Bug)
- ✅ Task 2: Fehlertext bei bf16 auf der CPU (Bug)
- ⬜ Task 3: Accelerate / MKL als optionales Feature
- ⬜ Task 4: GQA ohne `repeat_kv`, State-Keys vortransponiert
- ⬜ Task 5: Prefill ohne verworfene Arbeit im letzten Layer
- ⬜ Task 6: Letzter Layer nur an den Readout-Positionen
- ⬜ Task 7: Projektionen beim Laden zusammenlegen
- ⬜ Task 8: q statt Scores skalieren

## Bugs

### Task 1: `with_chunk_size` wirkt nicht

`qwen3_5::Backbone::recurrence` ruft `delta_rule_chunked(.., CHUNK)` mit der
Konstanten statt mit `self.chunk`, und `chunking_pays` rechnet ebenfalls mit
`CHUNK`. Das Feld wird gesetzt und nie gelesen.

Folge: `the_chunked_delta_rule_matches_the_sequential_one` läuft für
`[2, 8, 64]` dreimal mit 64 — die Rekursion von `unit_lower_inverse` bei kleinen
Blöcken ist ungetestet. Über die Ausgabe allein ist der Fehler nicht zu sehen,
weil jede Chunk-Größe dieselben Zahlen liefern *soll*. Was ihn zeigt: andere
Chunk-Größen summieren in anderer Reihenfolge, die Ergebnisse stimmen auf 1e-5
überein, aber nicht bitgenau. Stimmen sie bitgenau, hat die Größe die Rechnung
nicht erreicht.

Außerdem: der Doc-Kommentar von `delta_rule_chunked` beschreibt die Inverse
noch als `I + A + A² + …`, gebaut wird sie blockweise (`unit_lower_inverse`).

### Task 2: Fehlertext bei bf16 auf der CPU

`backend.rs`, `open_with`: dem String fehlt das `\` am Zeilenende, die Meldung
enthält mitten im Satz eine Reihe Leerzeichen.

## Optimierungen, nach erwartetem Nutzen

### Task 3: Accelerate (macOS) / MKL als optionales Feature

Die CPU-Matmuls laufen über candles `gemm`. Accelerate nutzt auf Apple Silicon
die AMX-Einheit und ist für f32-sgemm oft deutlich schneller. Opt-in wie
`metal` (`candle-core/accelerate`, `candle-nn/accelerate`), `scripts/test.sh`
auf Darwin mitnehmen. Messen: f32 packed (heute 4325 ms bzw. 2952 ms nach den
Fixes) und ob packed gegen separate weiterhin auf 0.00000 übereinstimmt.

### Task 4: GQA ohne `repeat_kv`, State-Keys vortransponiert

Jeder Branch-Pass kopiert pro Layer den ganzen State: `repeat_kv(past_k)`
(×2 bei 0.6B), `.transpose(2,3).contiguous()` und `repeat_kv(past_v)` — in
beiden Backbones. Bei 571 Tokens und 28 Layern grob einige hundert MB memcpy
pro Request, auch bei einem Cache-Hit. Dieselbe Fehlerklasse wie
`shared_matmul`, nur nicht zu Ende geführt.

Umbau: q als `[batch, kv_heads, repeats·len, dim]` falten und direkt gegen das
unwiederholte K/V multiplizieren; die Keys im `Prefix` beim Prefill schon
transponiert und contiguous ablegen. Die Transkriptionstests in
`tests/qwen3.rs`/`tests/qwen3_5.rs` müssen unverändert grün bleiben.

### Task 5: Prefill ohne verworfene Arbeit im letzten Layer

Ein Prefill behält nur K/V pro Layer, `run` rechnet für die State-Tokens aber
auch im letzten Layer Attention-Ausgabe, `o_proj`, MLP und die finale Norm.
Etwa 1/28 des Prefills, exakt einzusparen.

### Task 6: Letzter Layer nur an den Readout-Positionen

Im Packed- und im Branch-Pass braucht das MLP des letzten Layers (und die
finale Norm) nur die `<decide>`- und `</opt>`-Positionen. Wenige Prozent, mehr
Umbau als Task 5.

### Task 7: Projektionen beim Laden zusammenlegen

`gate_proj`+`up_proj` und `q`+`k`+`v` nach dem LoRA-Merge und vor der
Quantisierung zu je einer Projektion zusammenfügen. Weniger Kernel-Aufrufe,
`xs` wird einmal gelesen. Auf der CPU klein, auf Metal vermutlich mehr.

### Task 8: q statt Scores skalieren

`scores * scale` skaliert `[batch, heads, len, state+len]`; q vor dem Matmul zu
skalieren ist dieselbe Rechnung auf dem kleineren Tensor. Klein, aber trivial.

## Bewusst nicht angefasst

Die Prefix-Schwelle (384) kennt das Gerät nicht: auf Metal lohnt der Prefix
erst ab ~84 % Trefferquote, auf der CPU ab ~32 %. Den Default nicht ohne
Messung auf mehr als einem Modell ändern.
