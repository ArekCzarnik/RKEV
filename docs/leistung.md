# Geschwindigkeit und Quantisierung

Was die Hebel bringen, gemessen auf einem echten Checkpoint: Gerät, Präzision,
State-Prefix, Quantisierung. Und wie man es selbst nachmisst.

## Das Gerät

`metal` ist bewusst nicht Teil von `candle`: candles Metal-Backend zieht
Apple-Frameworks nach und baut auf keinem anderen System. Wo es fehlt, verweigert
`device("metal")` den Dienst statt auf die CPU zurückzufallen — ein stiller
Rückfall würde die Frage falsch beantworten, wenn der Grund fürs Fragen
Geschwindigkeit war:

```bash
cd rkev
cargo run --release --features metal --example measure -- \
    --base <base> --checkpoint <kev> --device metal
```

`decide`, `sanity`, `eval` und `measure` nehmen `--device`; die Genauigkeit folgt
dem Gerät, sofern `--dtype` nichts anderes sagt (bf16 auf GPU, f32 auf CPU, wie es
`kev.serve` wählt). `scripts/local.sh --device metal` setzt das Feature gleich mit.
Was das Gerät bringt, steht weiter unten gemessen.

## Quantisierung

Ein CPU-Pass ist bandbreitengebunden: entscheidend ist, wie viele Bytes an
Gewichten er lesen muss. `--quantise` packt die Projektionen in Blöcke — q4k
braucht etwa ein Siebtel von f32, q8_0 etwa ein Viertel:

```bash
cd rkev
cargo run --release --example measure -- \
    --base <base> --checkpoint <kev> --quantise q8_0
```

Vier Stufen: `q4k` (4.5 Bit), `q5k`, `q6k`, `q8_0` (8.5 Bit). `decide`, `sanity`,
`eval` und `measure` nehmen das Flag, `scripts/local.sh` reicht es weiter.

**Quantisiert wird nach dem LoRA-Merge, nie davor.** Der Merge bleibt exakt in f32,
und erst sein Ergebnis wird gerundet — ein vorquantisiertes Basismodell ließe sich
gar nicht mergen. Dicht bleiben die Embeddings, die Norms, die Faltung, die
Per-Head-Skalare und der Pointer-Head: die sind klein, und im Head steckt die
Kalibrierung.

**Was es kostet, musst du messen, nicht schätzen.** Die Antwort ist eine
kalibrierte Wahrscheinlichkeit; Rundung verschiebt sie. `measure --quantise`
druckt die Zeit **und** die größte Abweichung von f32 im selben Lauf, `eval
--quantise` die Trefferquote auf deinen Tickets. Erst beide Zahlen zusammen sind
eine Antwort.

Zwei Grenzen, beide als Weigerung statt als stiller Rückfall: Quantisierung läuft
nur mit **f32-Aktivierungen** (die ggml-Kerne wollen f32; f16 dazu würde zwei
Verluste vermischen, die man dann nicht mehr auseinanderhalten kann) und nur auf
der **CPU** (`QTensor::quantize` ist eine CPU-Routine; quantisiert auf Metal ist
hier nicht erprobt). Und die k-Quants packen 256 Gewichte pro Superblock: eine
Projektion, deren Zeile nicht durch 256 teilbar ist, wird namentlich abgelehnt mit
dem Hinweis auf q8_0, das 32 packt.

## Gemessen: kev-0.6b auf einem Apple M-Chip

Fünf Fragen über 571 State-Tokens, Median aus fünf Pässen, `kev-0.6b` über
`Qwen/Qwen3-0.6B-Base` (2026-09-24, macOS arm64, cargo 1.97):

| | f32 gepackt | f32 State einmal | f32 State im Cache | f16 State einmal |
|---|---|---|---|---|
| CPU | 2952 ms | 3681 ms | 1394 ms | 2752 ms |
| Metal | 605 ms | 998 ms | **529 ms** | 924 ms |
| *CPU, vor den Fixes* | *4325 ms* | *12416 ms* | *9112 ms* | *9484 ms* |
| *Metal, vor den Fixes* | *874 ms* | *3403 ms* | *2632 ms* | *3315 ms* |

Von 4325 ms auf 529 ms für dieselbe Anfrage: **8.2×**, davon 4.9× durch Metal und der
Rest durch die zwei Fehler unten.

Fünf Befunde, drei davon gegen die Erwartung:

- **Metal ist rund fünfmal schneller** als die CPU und bleibt es auch nach den Fixes:
  605 gegen 2952 ms gepackt, 529 gegen 1394 ms mit Cache-Treffer. f16 bringt dort
  nichts (1.08×, und 0.0032 Abweichung), auf der CPU 1.34×.
- **Auf Metal ist die Prefix-Wette eine andere.** Ein Fehlschlag kostet dort 65 %
  (998 gegen 605 ms), ein Treffer spart nur 13 % (529 ms) — die Schwelle von 384
  Tokens lohnt sich erst ab etwa 84 % wiederkehrender States, auf der CPU schon ab
  32 %. Wer auf einer GPU mit lauter neuen Dokumenten arbeitet, setzt
  `with_prefix_min_tokens(usize::MAX)` und fährt den gepackten Pass.
- **Ein gebatchter Pass rechnete jede Projektion pro Zeile neu.** candles
  `broadcast_matmul` verteilt nicht die Eingabe, sondern **materialisiert die
  Gewichtsmatrix für jede Batch-Zeile** — bei fünf Fragen also fünf Kopien von 8 MB
  pro Projektion, pro Layer, pro Pass. Isoliert gemessen (eine 2048×1024er
  Projektion über 5×45 Zeilen): **93.8 ms gegen 17.8 ms**, Faktor 5.3. `linear`
  faltet die Batch-Dimension jetzt in die Zeilen, so wie der quantisierte Kern es
  von sich aus tut. Betroffen war alles, was Zeilen batcht: die Zweige hinter einem
  Prefix, `system_one_batch_blocking` (jetzt 1.22× statt 0.96×), und auf einer
  hybriden Basis **jeder** Pass. Auch der gepackte Pfad zahlte einmal pro Projektion,
  weil ein rangzwei-Gewicht für einen dreirangigen Eingang ohnehin gebroadcastet
  wurde — daher fielen auch seine 4325 ms auf 2952 ms.
- **Der State-Prefix zahlt sich aus, sobald ein State wiederkommt** — und genau das
  hatte der Fehler oben verdeckt. Vorher kostete er das 2.9-Fache, was mich kurzzeitig
  die Voreinstellung abschalten ließ; mit einem gemm statt fünf steht es so: 2952 ms
  gepackt, 3681 ms bei einem Fehlschlag, **1394 ms bei einem Treffer**. Ein neuer
  State kostet also ein Viertel mehr, ein wiederkehrender die Hälfte weniger — die
  Wette lohnt ab etwa einem Drittel wiederkehrender States. Deshalb steht die
  Schwelle wieder auf den 384 Tokens der Python-Seite; `0` prefillt alles.
- **Quantisierung zahlt sich nicht aus.** q8_0 ist neutral (1.05×), q6k und q4k sind
  *langsamer* (0.64× und 0.76×) — candles k-Quant-Kerne schlagen auf Apple-Silizium
  den dichten f32-Pfad nicht. Dazu die Kosten an den Antworten: q8_0 0.077, q6k
  0.046, q4k **0.311** größte Abweichung von f32. Ein um 0.31 verschobener Wert ist
  für eine kalibrierte Wahrscheinlichkeit unbrauchbar.

Was durchgehend hielt: **7 von 7 eindeutigen Fällen**, in jeder Präzision und jeder
Quantisierungsstufe — der Argmax überlebt, die Kalibrierung nicht. Und die
Orientierung des Pointer-Heads ist damit empirisch belegt: **0 von 7 mit
vertauschten Projektionen.**

Die Zahl für f16 in der Präzisions-Sektion unten stammt von den Fixtures (0.01); auf
echten Gewichten sind es **0.03** auf der CPU und 0.0037 auf Metal. Der dichte
f32-Pfad auf der CPU ist der exakte: dort stimmen gepackter und getrennter Pass auf
0.00000 überein. Quantisiert und auf Metal weichen sie um 0.003 ab — Assoziativität
der Fließkommaaddition, nicht ein Fehler im Layout.

## Alles auf einmal messen

`scripts/sweep.sh` fährt einen Checkpoint durch jeden Pfad, den er hat, und legt
ein Protokoll an, das man weitergeben kann:

```bash
scripts/sweep.sh --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b
```

CPU dicht, dann q8_0, q6k und q4k, dann dieselben Läufe durch `sanity` — weil eine
Beschleunigung, die die Entscheidungen beschädigt, keine ist. Danach Metal, dicht
(quantisiert ist ein CPU-Pfad, dafür gibt es absichtlich keine Metal-Zeile). Nichts
darin ist fatal: ein Lauf, der abbricht, wird vermerkt und der Rest läuft weiter —
eine fehlende Metal-Op soll nicht die CPU-Zahlen kosten.

Am Ende sammelt es die Verhältniszeilen und trennt zwei Dinge, die leicht
verwechselt werden: **„did not finish"** ist ein Lauf, der nicht durchkam (meist
eine Weigerung, die selbst schon die Antwort ist), **„not convinced"** ein Lauf, der
durchkam und *nein* sagte — der Checkpoint hat die eindeutigen Fälle verfehlt. Das
Zweite ist bei echten Gewichten das Ergebnis, nicht ein Fehler.

`--skip-metal`, `--skip-cpu`, `--repeat`, `--words`, `--quantise "q8_0 q4k"` und
`--out` stellen ein, was läuft; `--help` zeigt alles. Das Protokoll heißt
`rkev-measurements.txt` und ist in der `.gitignore`.

---

[← zurück zur README](../README.md)
