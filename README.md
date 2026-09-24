# RKEV 1.0p

### RKev 1.0p Kevs Entscheidungsmodelle in Rust
-> Prompt, Forward-Pass und Readout laufen in Rust im eigenen Prozess <-

RKev führt Kevs Entscheidungsmodelle direkt in Rust im eigenen Prozess aus: ohne zusätzlichen Server, ohne API-Gateway und ohne laufende API-Kosten. Alles läuft vollständig lokal.

Das Prinzip ist einfach: Text (oder JSON) + Fragestellung rein → Wahrscheinlichkeiten raus. Unterstützt werden Ja/Nein-Entscheidungen, die Auswahl aus mehreren Optionen sowie Bewertungen auf einer Skala.
Dabei werden keine Antwort-Tokens generiert. Die Entscheidung erfolgt über einen Pointer-Head auf den Hidden States der Qwen-Basis, in die der LoRA-Adapter des Checkpoints eingerechnet ist.

RKev unterstützt Qwen3 ebenso wie das hybride Qwen3.5 mit Gated DeltaNet. Dazu kommen ein CLI für direkte Anfragen und ein Eval-Runner, mit dem sich eigene gelabelte Tickets auswerten lassen — mit Trefferquote pro Sicherheitsband, aus dem eine Schwelle fürs automatische Routing folgt.

## Was RKev macht

Die Fragen teilen sich den State, können einander aber nicht lesen. Drei Typen:

| Typ | Frage | Antwort |
|---|---|---|
| `noul` | ja/nein | `noul`: die Wahrscheinlichkeit für ja |
| `choice` | eine von 1–255 benannten Optionen | `choice`, `confidence`, die ganze Verteilung |
| `score` | eine geordnete Skala aus 1–255 Stufen | `score` (mittlerer Stufenindex), `confidence`, Verteilung, Legende |

Kev **generiert nichts**. Ein Checkpoint ist ein LoRA-Adapter über einer
Qwen-Basis plus ein kleiner *Pointer-Head*: die Antwort entsteht daraus, dass die
Hidden States der Options-Enden gegen den `<decide>`-Token bewertet werden. Genau
deshalb kann keine Text-Generierungs-Engine diese Modelle bedienen — gebraucht
werden Hidden States, nicht Logits.

## Schnellstart

Ein Skript holt einen Checkpoint (mit `curl` — das `hf`-CLI ist selbst Python) und
prüft danach alles, woran ein Checkpoint allein gemessen werden kann:

```bash
scripts/local.sh --fetch jaredpalmer/kev-0.6b
```

Danach beantwortest du Anfragen direkt:

```bash
cd rkev
cargo run --release --example decide -- \
    --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b \
    --state "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
```

Das ist ein gemessener Lauf von `kev-0.6b` in f32 auf einer Apple-CPU, kein
Beispielbild:

```text
attention-only base in F32, loaded in 1.0s
department     billing      0.51   confidence 0.26
               shipping     0.25
               returns      0.24
escalate       no           0.41
frustration    Frustrated   level 1.04   confidence 0.97
               Calm         0.01
               Frustrated   0.95
               Very angry   0.05
               101 tokens in, 1671 ms
```

Das Ticket nennt absichtlich drei Abteilungen gleichzeitig; bei eindeutigen
Tickets sitzt dieses Modell auf 1.00.

`decide` nimmt auch `--request anfrage.json` (wie du es sonst POSTen würdest,
mehrfach = Batch), `--questions` zu einem `--state`, `--lines` für einen State pro
Zeile von stdin und `--json` für genau die Antwort, die der Server geschickt hätte.
Ob die Checkpoints auf **deutschen** Tickets genauso zuverlässig sind, messt
`examples/deutsch` — siehe [docs/eval.md](docs/eval.md#auf-deutsch).

## Als Bibliothek

```toml
[dependencies]
rkev = { path = "rkev" }
```

```rust
use rkev::{pointer_head, Backend, Choice, LocalEngine, Noul, SystemOneRequest};

// Basismodell plus Checkpoint; welche Architektur nötig ist, steht in dessen config.json.
let backend = Backend::open(base_dir, Some(checkpoint_dir))?;
// Die zweite Hälfte eines Checkpoints: die Projektionen aus head.pt und die
// Temperatur, mit der er kalibriert wurde.
let head = pointer_head(&checkpoint_dir.join("head.pt"))?;
let engine = LocalEngine::new(backend, head);

let request = SystemOneRequest::new("Schuhe zwei Wochen zu spät und in der falschen Größe.")
    .ask(
        "department",
        Choice::new("Welches Team soll das übernehmen?")
            .option("returns", "Umtausch, Rückgabe, falsche oder beschädigte Ware")
            .option("shipping", "Lieferstatus, Verzug, verlorene Pakete")
            .option("billing", "Abbuchungen, Rechnungen, Zahlungsprobleme"),
    )
    .ask("escalate", Noul::new("Braucht das dringend einen Menschen?"));

let response = engine.system_one_blocking(&request)?;   // oder .system_one(..).await
```

Ein asynchroner Aufrufer nimmt `engine.system_one(&request).await` — das schiebt
den Pass vom Runtime-Thread weg, weil ein Forward-Pass CPU-gebunden ist.
`LocalEngine` ist billig zu klonen, und Klone teilen das eine geladene Modell.

## Features

| Feature | Was es bringt | Kosten |
|---|---|---|
| `candle` (Standard) | das Modell selbst: beide Qwen-Generationen | candle 0.9, tokenizers, zip |
| `local` | Prompt, Token-Layout, Pointer-Head, `LocalEngine`; `Forward` bleibt dir | tokio (nur `spawn_blocking`) |
| — | die Wire-Format-Typen, die Fehler, die `SystemOne`-Naht | nichts |
| `metal` | candles Metal-Backend für eine Apple-GPU; nur auf macOS baubar | Apple-Frameworks |

Ohne jedes Feature (`--no-default-features`) bleibt also genau das, was ein
Aufrufer braucht, um mit etwas anderem zu reden — oder um eine Aufzeichnung zu
halten. MSRV ist 1.75 ohne `candle`, mit `candle` dessen eigener Wert.

`metal` ist bewusst nicht Teil von `candle` und nur auf macOS baubar; wo es fehlt,
verweigert `device("metal")` den Dienst statt still auf die CPU zurückzufallen.
Details und die gemessenen Zahlen: [docs/leistung.md](docs/leistung.md).

## Beispiele

Alle in `rkev/examples/`:

| Beispiel | Wofür | Braucht |
|---|---|---|
| `decide` | Anfragen beantworten — der Server-Job im eigenen Prozess | Checkpoint |
| `sanity` | prüft die Engine gegen sich selbst und gegen eindeutige Fälle | Checkpoint |
| `measure` | f16 gegen f32, Prefix-Cache, Chunking, Batching — mit Kontrollzeilen | Checkpoint |
| `parity` | vergleicht jede Wahrscheinlichkeit mit einer aufgezeichneten Server-Antwort | Aufzeichnung |
| `deutsch` | ein deutsches Ticket mit allen drei Fragetypen, und mit `--vergleich` dieselben Inhalte auf englisch daneben | Checkpoint |
| `eval` | Trefferquote und Kalibrierung auf deinen eigenen beschrifteten Tickets | Checkpoint + Records |

## Skripte

```bash
scripts/test.sh                                 # fmt, clippy, Tests, Feature-Matrix
scripts/local.sh                                # die Offline-Suite allein
scripts/local.sh --fetch jaredpalmer/kev-0.6b   # Checkpoint holen, dann alles prüfen
scripts/local.sh --checkpoint <dir> --measure   # dazu die Zeitmessungen
scripts/local.sh --checkpoint <dir> \
    --questions q.json --records tickets.jsonl  # dazu deine eigenen Tickets
scripts/sweep.sh --base <dir> --checkpoint <dir>     # CPU und Metal durchmessen
scripts/parity.sh --base <dir> --checkpoint <dir>    # gegen einen Server aufzeichnen
scripts/parity.sh --check-only --base <dir> --checkpoint <dir>   # und offline nachprüfen
```

`scripts/local.sh --help` listet den Rest. `KEV_HF` zeigt die Downloads auf einen
Spiegel (oder auf einen `file://`-Baum).

## Stand und Grenzen

Geprüft: beide Backbones (attention-only Qwen3 und das hybride Qwen3.5 mit Gated
DeltaNet), beide Forward-Pässe gegen Transkriptionen der Hugging-Face-Modelle,
92 Tests. Ein echter Checkpoint lädt und antwortet sinnvoll — `jaredpalmer/kev-0.6b`
trifft alle sieben eindeutigen Fälle von `sanity`, und mit vertauschten
Pointer-Head-Projektionen keinen einzigen.

Offen ist genau eines: **Parität mit der Referenz.** Dass die Zahlen plausibel und
untereinander konsistent sind, heißt nicht, dass sie dieselben sind. Das ist die
einzige Stelle, an der der Python-Server noch gebraucht wird, und auch dort genau
einmal — danach sind die Aufzeichnungen Dateien.

Dazu drei Grenzen: auf einer CPU gibt es in candle kein bf16-Matmul (f16 ist die
reduzierte Präzision dort), ein echter **hybrider** Checkpoint wurde noch nicht
geladen, und gelaufen sind bisher CPU und Metal — sonst kein Gerät.

Mehr dazu in [docs/pruefungen.md](docs/pruefungen.md).
## Mehr im Detail


| | |
|---|---|
| [docs/ablauf.md](docs/ablauf.md) | Wie eine Antwort entsteht: Prompt, Token-Layout, Forward-Pass, Pointer-Head — und was das deutsche Beispiel davon benutzt |
| [docs/leistung.md](docs/leistung.md) | Gemessene Zahlen: Metal gegen CPU, f16, State-Prefix, Quantisierung, und wie man selbst messt |
| [docs/eval.md](docs/eval.md) | Trefferquote und Kalibrierung auf eigenen beschrifteten Tickets, mit Schwelle für CI |
| [docs/pruefungen.md](docs/pruefungen.md) | Woran der Port festgemacht ist, und wie ein falsch geladener Checkpoint laut wird |
