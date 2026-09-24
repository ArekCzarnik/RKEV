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
- ✅ Task 4: GQA ohne `repeat_kv`, State-Keys vortransponiert
- ✅ Task 5: Prefill ohne verworfene Arbeit im letzten Layer
- ✅ Task 6: Letzter Layer nur an den Readout-Positionen
- ⬜ Task 7: Projektionen beim Laden zusammenlegen
- ⬜ Task 8: q statt Scores skalieren
- ⬜ Task 9: Maske nur auf den Branch-Teil der Scores
- ⬜ Task 10: Qwen3-Prefix-Messung misst den Packed-Pass

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

**Erledigt.** `weights::grouped_attention` ersetzt die Attention-Rümpfe beider
Backbones; `repeat_kv` ist weg, `state_keys` transponiert die Keys einmal beim
Prefill. Gemessen isoliert, eine Attention-Schicht in kev-0.6b-Shapes (16 über
8 Heads, 128 breit, 571 State-Tokens, 5×45 Branch-Zeilen), Linux-Container
aarch64, dreimal: 80–90 ms → 65–69 ms, **~1,25x, bitgleich**
(`what_reading_the_state_unrepeated_is_worth`). Auf einem echten Checkpoint
noch nicht gemessen. Zwei absichtliche Brüche (falsche Head-Zuordnung, Keys
nicht transponiert) lassen die Transkriptions- bzw. Prefix-Tests beider
Backbones fehlschlagen.

Aufteilung danach, pro Schicht: q·State-Keys 16 ms, Gewichte·State-Values
20 ms, Maske + Skalierung 11 ms, Softmax 5 ms, q·Branch-Keys 4 ms, Transponieren
der State-Keys 1,3 ms. Der Gewinn kam also aus den Wiederholungen, nicht aus
dem Transponieren — und Maske + Skalierung sind der nächste sichtbare Posten
(Task 8, Task 9).

### Task 5: Prefill ohne verworfene Arbeit im letzten Layer

Ein Prefill behält nur K/V pro Layer, `run` rechnet für die State-Tokens aber
auch im letzten Layer Attention-Ausgabe, `o_proj`, MLP und die finale Norm.
Etwa 1/28 des Prefills, exakt einzusparen.

**Erledigt.** `run` ist in `stack` (die Layer) und die finale Norm geteilt;
Prefills rufen `cache`, das im letzten Layer nach den Keys und Values (bzw.
dem rekurrenten State) aufhört. Ein rekurrenter letzter Layer läuft voll, nur
sein MLP fällt weg. Gemessen mit `what_a_prefill_costs` (breite Hybrid-Fixture,
511 Tokens, **2 Layer**), je dreimal: 630–700 ms → 525–610 ms. Bei zwei Layern
ist das eine halbe Schicht von zweien — auf 28 Layern bleiben davon
erwartungsgemäß nur ein paar Prozent, gemessen ist das nicht.

Dabei fiel auf: alle Hybrid-Fixtures enden auf `full_attention`, ein
rekurrenter letzter Layer lief in keinem Test.
`a_prefix_ending_in_a_recurrent_layer_changes_no_answer_either` tauscht die
beiden Layer und ist der einzige Test, der einen Fehler in diesem Zweig findet
(geprüft durch absichtliches Brechen); der Pfad über `keys_values` wird in
beiden Backbones von den Prefix-Tests gefangen.

### Task 6: Letzter Layer nur an den Readout-Positionen

Im Packed- und im Branch-Pass braucht das MLP des letzten Layers (und die
finale Norm) nur die `<decide>`- und `</opt>`-Positionen. Wenige Prozent, mehr
Umbau als Task 5.

**Erledigt.** `forward` und `forward_from_batch` beider Backbones nehmen die
Readout-Positionen und geben nur diese zurück; `Backend::pick` ist weg. Im
letzten Layer laufen Keys und Values über alle Tokens, Query, Attention-Ausgabe,
`o_proj`, MLP und finale Norm nur an den Readout-Positionen (`select_rows`,
`Wanted` in `weights.rs`; `grouped_attention` nimmt dafür weniger Query-Zeilen
als Keys). Ein rekurrenter letzter Layer läuft voll und wird vor dem MLP
ausgedünnt.

Gemessen, je dreimal, Toy-Fixtures mit **2 Layern**: Packed-Pass 17–18 ms →
10–11 ms, Hybrid eine Zeile pro Frage 51 ms → 27–32 ms. Nur Branches auf der
breiten Fixture: 220–244 → 218–272 ms, also nichts — die Branch-Zeilen sind
kurz, und fast jedes ihrer Tokens ist eine Readout-Position. Auf 28 Layern ist
für den Packed-Pass etwa ein Layer minus K/V zu erwarten, ein paar Prozent;
gemessen ist das nicht.

Prüfung: drei absichtliche Brüche (rekurrenter letzter Layer liest `normed`
statt seiner Ausgabe, `select_rows` ignoriert die Batch-Zeile, Residuum aus
`normed`) werden gefangen. Den ersten fängt nur der neue Test
`a_checkpoint_ending_in_a_recurrent_layer_matches_the_transcription` — dafür
nimmt `reference_hidden` jetzt die Layer-Typen als Parameter, und `reversed()`
baut die Fixture mit getauschten Layern.

### Task 7: Projektionen beim Laden zusammenlegen

`gate_proj`+`up_proj` und `q`+`k`+`v` nach dem LoRA-Merge und vor der
Quantisierung zu je einer Projektion zusammenfügen. Weniger Kernel-Aufrufe,
`xs` wird einmal gelesen. Auf der CPU klein, auf Metal vermutlich mehr.

### Task 8: q statt Scores skalieren

`scores * scale` skaliert `[batch, heads, len, state+len]`; q vor dem Matmul zu
skalieren ist dieselbe Rechnung auf dem kleineren Tensor. Klein, aber trivial.

Nicht bitgleich, sobald `1/sqrt(dim)` keine Zweierpotenz ist (128: nein, 64:
ja) — also gegen die Toleranzen der Transkriptionstests prüfen, nicht auf
Gleichheit.

### Task 9: Maske nur auf den Branch-Teil der Scores

In einem Branch-Pass ist die Maske über den ganzen State-Teil null
(`branch_batch_mask`: jeder Branch darf den ganzen State lesen). Trotzdem wird
sie per Broadcast über `[batch, heads, len, state+len]` addiert — gemessen
zusammen mit der Skalierung 11 ms pro Schicht bei 571 State-Tokens. Nur auf den
`[.., len, len]`-Branch-Teil angewendet, vor dem `cat`, ist es exakt dasselbe
und etwa ein Dreizehntel der Elemente.

### Task 10: Qwen3-Prefix-Messung misst den Packed-Pass

`how_much_the_prefix_saves` in `tests/qwen3.rs` fragt mit 241 State-Tokens,
unter der Schwelle von 384, ohne `with_prefix_min_tokens(0)` — die Zeilen
„cache hit“ und „new state, prefilled“ messen beide den Packed-Pass. Genau die
Falle, vor der `examples/measure.rs` sich schützt.

## Bewusst nicht angefasst

Die Prefix-Schwelle (384) kennt das Gerät nicht: auf Metal lohnt der Prefix
erst ab ~84 % Trefferquote, auf der CPU ab ~32 %. Den Default nicht ohne
Messung auf mehr als einem Modell ändern.
