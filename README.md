# kevexample — Kev in Rust

Eine vollständige lokale Inferenz-Engine für
[Kev](https://github.com/jaredpalmer/kev), die kleinen Entscheidungsmodelle, die
TypeSafe's [System One](https://docs.typesafe.ai/api)-API sprechen.

Prompt, Forward-Pass und Readout laufen in Rust im eigenen Prozess — **ohne
Server und ohne Python**. Kev selbst ist Python; gebraucht wird es hier nur noch
für eine Sache, nämlich den Vergleich der Zahlen mit der Referenz (siehe *Stand
und Grenzen*).

Das einzige Crate liegt in [`kev-client/`](kev-client/); das Wurzelverzeichnis ist
kein Cargo-Workspace, `cargo`-Befehle laufen also aus `kev-client/`. Die
ausführliche, englische Dokumentation des Crates steht in
[`kev-client/README.md`](kev-client/README.md).

## Was Kev macht

Du schickst einen **State** — ein Ticket, ein Dokument, beliebigen Text oder JSON
— und dazu eine Menge **Fragen**. Zurück kommen Wahrscheinlichkeiten statt eines
einzelnen Labels. Die Fragen teilen sich den State, können einander aber nicht
lesen.

Drei Fragetypen:

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

## Schnellstart, ohne Python und ohne Server

Ein Skript holt einen Checkpoint (mit `curl` — das `hf`-CLI ist selbst Python) und
prüft danach alles, was ohne Server prüfbar ist:

```bash
scripts/local.sh --fetch jaredpalmer/kev-0.6b
```

Danach beantwortest du Anfragen direkt:

```bash
cd kev-client
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
Tickets sitzt dieses Modell auf 1.00. Ob die Checkpoints auf deutschen Texten
genauso zuverlässig sind, ist hier nicht gemessen — das findest du mit `--lines`
an deinen eigenen Tickets am schnellsten heraus.

Weitere Eingabeformen: `--request anfrage.json` (genau das, was du sonst POSTen
würdest, mehrfach = Batch), `--questions fragen.json` zu einem `--state`,
`--lines` für einen State pro Zeile von stdin (Modell lädt einmal, Cache bleibt
warm), `--json` für exakt die Antwort, die der Server geschickt hätte. Diagnose
geht auf stderr, Antworten auf stdout.

## Als Bibliothek

```toml
[dependencies]
kev-client = { path = "kev-client" }
```

```rust
use kev_client::{pointer_head, Backend, Choice, LocalEngine, Noul, SystemOneRequest};

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

Ohne jedes Feature (`--no-default-features`) bleibt also genau das, was ein
Aufrufer braucht, um mit etwas anderem zu reden — oder um eine Aufzeichnung zu
halten. MSRV ist 1.75 ohne `candle`, mit `candle` dessen eigener Wert.

## Beispiele

Alle in `kev-client/examples/`:

| Beispiel | Wofür | Braucht |
|---|---|---|
| `decide` | Anfragen beantworten — der Server-Job im eigenen Prozess | Checkpoint |
| `sanity` | prüft die Engine gegen sich selbst und gegen eindeutige Fälle | Checkpoint |
| `measure` | f16 gegen f32, Prefix-Cache, Chunking, Batching — mit Kontrollzeilen | Checkpoint |
| `parity` | vergleicht jede Wahrscheinlichkeit mit einer aufgezeichneten Server-Antwort | Aufzeichnung |

## Skripte

```bash
scripts/test.sh                                 # fmt, clippy, Tests, Feature-Matrix
scripts/local.sh                                # die Offline-Suite allein
scripts/local.sh --fetch jaredpalmer/kev-0.6b   # Checkpoint holen, dann alles prüfen
scripts/local.sh --checkpoint <dir> --measure   # dazu die Zeitmessungen
```

`scripts/local.sh --help` listet den Rest. `KEV_HF` zeigt die Downloads auf einen
Spiegel (oder auf einen `file://`-Baum).

## Tests

Alle offline, keiner braucht einen Server oder Gewichte:

```bash
cd kev-client
cargo test                                    # 83
cargo test --no-default-features --features local   # 50
cargo test --no-default-features              # 14
```

Wo die Erwartungen herkommen, ist der Punkt: beide Forward-Pässe werden gegen
Transkriptionen von Hugging Faces `modeling_qwen3.py` und `modeling_qwen3_5.py`
verglichen, Hidden-Unit für Hidden-Unit, und `tests/upstream_unit.rs` portiert
Kevs eigene `tests/test_unit.py`. Mit `KEV_TOKENIZER=<tokenizer.json>` kommen die
Prüfungen dazu, die ein echtes Qwen-Vokabular brauchen.

## Stand und Grenzen

Was geprüft ist:

- Beide Backbones: attention-only Qwen3 **und** das hybride Qwen3.5 (Gated
  DeltaNet, gated Attention, zero-centred Norms, partielles Rotary).
- Ein echter Checkpoint lädt und antwortet sinnvoll: `jaredpalmer/kev-0.6b` über
  `Qwen/Qwen3-0.6B-Base` in f32 auf einer Apple-CPU trifft alle sieben
  eindeutigen Fälle von `sanity`, liest `head.pt` samt Temperatur und
  `option_isolation`, und die verschiedenen Pfade stimmen auf fünf Dezimalen
  überein.
- Geschwindigkeit: State-Prefix mit Cache, gebatchte Zweige, gebatchte Prefills,
  Delta-Rule in Chunks von 64 Tokens. Alles exakt — die Tests verlangen
  identische Antworten.

Was offen ist:

- **Parität mit dem Python-Server ist nicht geprüft.** Dass die Zahlen plausibel
  und untereinander konsistent sind, heißt nicht, dass sie dieselben sind. Dafür
  braucht es den Server einmal; `examples/parity.rs` ist dieser Vergleich.
- Auf einer CPU gibt es in candle kein bf16-Matmul, also wird bf16 dort mit einer
  Meldung abgelehnt statt tief in einer Projektion zu scheitern; f16 ist die
  reduzierte Präzision, die eine CPU kann.
- Vom hybriden Qwen3.5-Pfad ist die Arithmetik geprüft, aber noch **kein echter
  hybrider Checkpoint geladen** — der echte Lauf war ein attention-only Modell.
- Nichts ist quantisiert, und außer der CPU ist kein Gerät gelaufen.

## Herkunft

Kev selbst ist ein eigenes Projekt: <https://github.com/jaredpalmer/kev>
(Apache-2.0). Die Regeln, die Prompt und Readout hier umsetzen, sind aus dessen
Python übernommen, nicht erfunden — `kev/api.py` und `kev/model.py`. Änderungen
daran folgen der Python-Seite, nie der eigenen Meinung.
