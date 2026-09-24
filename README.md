# RKEV 1.0p

### RKev 1.0p Kevs Entscheidungsmodelle in Rust
-> Prompt, Forward-Pass und Readout laufen in Rust im eigenen Prozess <-

RKev führt Kevs Entscheidungsmodelle direkt in Rust im eigenen Prozess aus: ohne zusätzlichen Server, ohne API-Gateway und ohne laufende API-Kosten. Alles läuft vollständig     
lokal.

Das Prinzip ist einfach: Text (oder JSON) + Fragestellung rein → Wahrscheinlichkeiten raus. Unterstützt werden Ja/Nein-Entscheidungen, die Auswahl aus mehreren Optionen sowie Bewertungen auf einer Skala.

Dabei werden keine Antwort-Tokens generiert. Die Entscheidung erfolgt über einen Pointer-Head auf den Hidden States der Qwen-Basis, in die der LoRA-Adapter des Checkpoints      
eingerechnet ist.

RKev unterstützt Qwen3 ebenso wie das hybride Qwen3.5 mit Gated DeltaNet. Dazu kommen ein CLI für direkte Anfragen und ein Eval-Runner, mit dem sich eigene gelabelte Tickets    
auswerten lassen — mit Trefferquote pro Sicherheitsband, aus dem eine Schwelle fürs automatische Routing folgt.    
--
Das einzige Crate liegt in [`rkev/`](rkev/); das Wurzelverzeichnis ist
kein Cargo-Workspace, `cargo`-Befehle laufen also aus `rkev/`. Die
ausführliche, englische Dokumentation des Crates steht in
[`rkev/README.md`](rkev/README.md).

## Was RKev macht

Du schickst einen **State** , ein Ticket, ein Dokument, beliebigen Text oder JSON
, und dazu eine Menge **Fragen**. Zurück kommen Wahrscheinlichkeiten statt eines
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
deshalb kann keine Text-Generierungs-Engine diese Modelle bedienen , gebraucht
werden Hidden States, nicht Logits.

## Vom Ticket zur Antwort: der Weg von RKev nach Qwen

Vier Schritte, vier Module. Jeder Schritt ist im Code an einer Stelle, und die
Reihenfolge ist immer dieselbe , es gibt keine Schleife und keine Generierung.

### 1. Anfrage → Text (`rkev/src/prompt.rs`)

Der State wird zu Text. Ein String bleibt, wie er ist; ein JSON-Objekt wird zu
`schlüssel: wert` pro Zeile, eine Liste zu `- eintrag`, verschachteltes eingerückt.
Bei Skalaren zählt die Schreibweise: `True` und `False` statt `true`/`false`, und
`null` wird zu **nichts** , also `schlüssel: ` mit leerem Wert. Das ist kein
Geschmack, sondern Bedingung: die Referenz-Implementierung (siehe *Herkunft*) baut
genau diesen Text, und ein anderer Text ist eine andere Antwort.
`tests/upstream_unit.rs` hält sechs dieser Ausgaben an deren eigenen Testfällen
fest.

Jede Frage wird zu Anweisungen plus einer Liste von **Optionstexten**, jeweils
`name: beschreibung`. Die drei Fragetypen unterscheiden sich nur darin, was die
Optionen sind:

| Typ | Optionen im Prompt | Schlüssel in der Antwort |
|---|---|---|
| `noul` | zwei, `no` und `yes` (plus deren Beschreibungen) | `false`, `true` , die Antwort ist die Wahrscheinlichkeit der zweiten |
| `choice` | deine Optionen, in deiner Reihenfolge | die Optionsnamen |
| `score` | die Stufenbeschreibungen, niedrigste zuerst | die Stufenindizes `0`, `1`, … |

### 2. Text → Tokens (`rkev/src/encode.rs`)

Fünf selten benutzte Qwen-Spezialtokens dienen als Trennzeichen, damit keine
Embedding-Zeilen dazukommen müssen , der LoRA-Adapter gibt ihnen ihre Bedeutung:

| Rolle | Token |
|---|---|
| `<state>` | `<\|fim_prefix\|>` |
| `<q>` | `<\|fim_middle\|>` |
| `<opt>` | `<\|box_start\|>` |
| `</opt>` | `<\|box_end\|>` |
| `<decide>` | `<\|fim_suffix\|>` |

Daraus wird ein Layout: der State einmal, danach pro Frage ein eigener Zweig.

```text
<state> …der State…
<q> Anweisungen <opt> Option 1 </opt> <opt> Option 2 </opt> … <decide>
<q> Anweisungen <opt> Option 1 </opt> … <decide>
```

Drei Dinge entstehen hier mit:

- **Die Maske.** Ein Token darf den State und seine eigene Frage lesen, sonst
  nichts. Deshalb braucht ein Backend `Pass::attends` und keine gewöhnliche
  kausale Maske , nur so können die Fragen einander nicht lesen.
- **Die Positionen.** Jeder Zweig fängt direkt hinter dem State wieder an. Der
  State wird also einmal verarbeitet, egal wie viele Fragen mitkommen.
- **Die Ableseplätze.** Pro Frage der Index ihres `<decide>` und die Indizes
  aller `</opt>`.

Außerdem wird Aufrufertext maskiert: `<|name|>` wird zu `<¦name¦>`. Ohne das
könnte ein State eine Frage aufmachen, denn ein Tokenizer erkennt seine
Spezialtokens auch mitten im Text. Der State wird auf 8192 Tokens gekürzt
(trainiert wurde auf 384 State-Tokens und 1024 für State plus einen Zweig).

### 3. Tokens → Hidden States (`src/backend.rs`, `src/qwen3.rs`, `src/qwen3_5.rs`)

Hier fängt Qwen an. `Backend` setzt drei Dinge zusammen:

1. den **Tokenizer** des Checkpoints (Truncation und Padding ausdrücklich aus,
   wie es transformers bei jedem Aufruf macht),
2. die **Basisgewichte** aus den safetensors, mit dem **LoRA-Adapter des
   Checkpoints hineingerechnet** , in f32, *bevor* in die Rechengenauigkeit
   gecastet wird (`src/weights.rs`),
3. das **Backbone**, das aus der `config.json` der Basis folgt: `qwen3.rs` für
   die reinen Attention-Modelle, `qwen3_5.rs` für die hybriden mit Gated
   DeltaNet.

Dann läuft **ein** Vorwärtspass. Keine KV-Cache-Schleife, kein Token für Token,
kein Vokabular-Head: der Pass hört bei den letzten Hidden States auf, und
zurückgegeben werden nur die an den Ableseplätzen.

Wie er läuft, hängt von der Basis ab:

- **Attention-only:** ein einziger Pass über das ganze Layout, mit der Maske aus
  Schritt 2 als additiver Maske und den Positionen als `position_ids`.
- **Hybrid (Qwen3.5):** welche Layer eine Rekurrenz sind, sagt `layer_types` in
  der `config.json` , bei den veröffentlichten Basen sind es drei von vier. Und
  eine Rekurrenz kann man nicht bitten, die Tokens einer anderen Frage zu
  überspringen: sie läuft die Tokens ab. Deshalb wird pro Frage eine eigene
  kausale Zeile gerechnet , State, dann ihr Zweig. Das ist exakt statt maskiert
  und genau das, was die Referenz dort auch tut.

Im Layout steht der State ohnehin nur einmal. Als **eigener** Pass, von dem die
Fragen dann fortsetzen, läuft er, wenn es sich lohnt: auf einer rekurrenten Basis
immer (sonst würde jede Frage den ganzen State erneut ablaufen), auf einer
attention-only ab 384 State-Tokens , darunter spart es nichts, weil der gepackte
Pass den State schon einmal rechnet. Fortgesetzt wird bei Attention über die
gespeicherten Keys und Values, bei der Rekurrenz über Zustandsmatrix und
Faltungsfenster. Die letzten vier Zustände bleiben über Anfragen hinweg im Cache,
nach ihren Tokens geschlüsselt, wie es `kev.serve` auch macht.

### 4. Hidden States → Antwort (`rkev/src/readout.rs`)

Der **Pointer-Head** ist die zweite Hälfte eines Checkpoints und liegt in
`head.pt`. Für jede Frage:

```text
q      = W_q · h(<decide>)                     (Query)
k_i    = W_k · h(</opt> der Option i)          (Key je Option)
logit_i = ⟨q, k_i⟩ · 1/√d / T                  (T = Kalibriertemperatur)
p      = softmax(logits)                        (in f32)
```

`d` ist die Pointer-Dimension, `T` die Temperatur, die im `head.pt` mitgespeichert
ist. Aus der Verteilung wird die Antwort:

- `noul`: `p(yes)`, und keine Confidence , die Wahrscheinlichkeit *ist* die
  Antwort.
- `choice`: die wahrscheinlichste Option, Confidence `(p_max − 1/K)/(1 − 1/K)`.
- `score`: der mittlere Stufenindex, Confidence `1 − E|Stufe − Modus|/(L−1)`.

Alles auf vier Dezimalen gerundet, wie in der Referenz. Die Confidence ist ein Maß
für die *Form* der Verteilung, keine gemessene Trefferquote , TypeSafes
API-Dokumentation definiert sie nicht, diese Formeln sind die der Referenz, und
`tests/upstream_unit.rs` hält sie an deren eigenen Testfällen fest.

```text
SystemOneRequest
   │  prompt.rs    State → Text, Fragen → Optionstexte
   ▼
Record
   │  encode.rs    fünf Trennzeichen, Maske, Positionen, Ableseplätze
   ▼
Tokens + Maske
   │  backend.rs   Tokenizer, Basis + LoRA, ein Pass, kein Vokabular-Head
   │  qwen3.rs / qwen3_5.rs
   ▼
Hidden States an <decide> und </opt>
   │  readout.rs   Pointer-Head, Softmax, Rundung
   ▼
SystemOneResponse
```

### Was das deutsche Beispiel davon benutzt

`rkev/examples/deutsch.rs` geht genau diesen Weg, einmal pro Anfrage:

| Im Beispiel | Was dahinter passiert |
|---|---|
| `Backend::open(&basis, checkpoint)` | Tokenizer laden, Basis + LoRA mergen, Backbone aus der `config.json` wählen (Schritt 3) |
| `pointer_head(&kopf)` | `head.pt` lesen: die zwei Projektionen **und** die Temperatur (Schritt 4) |
| `option_isolation(&kopf)` | fragt den Checkpoint, für welches Layout er trainiert wurde; das falsche zu servieren wäre ein leise anderer Prompt |
| `LocalEngine::new(…)` | verbindet beide Hälften; `Clone` ist billig, das Modell wird nicht zweimal geladen |
| `SystemOneRequest::new(text)` | der State , hier ein deutscher String, also geht er unverändert in Schritt 1 |
| `.ask("abteilung", Choice::new(…).option("versand", …))` | wird zu `<q> Anweisungen <opt> versand: Lieferstatus… </opt> … <decide>` |
| `.ask("eskalation", Noul::new(…))` | wird zu zwei Optionen, `no` und `yes` |
| `.ask("verärgerung", Score::new(…).level("ruhig"))` | wird zu einer Option pro Stufe, niedrigste zuerst |
| `motor.system_one_blocking(…)` | Schritte 1–4 in einem Aufruf, blockierend (für async gibt es `.system_one(..).await`) |
| `Answer::Choice { probabilities, .. }` | die Verteilung aus Schritt 4, in der Reihenfolge der Optionen |

Zwei Dinge sind deshalb im Beispiel deutsch und nicht nur der Ticket-Text:

- **Die Optionsnamen.** Sie stehen als `name: beschreibung` im Prompt (Schritt 1)
  und sind damit Teil der Sprache, kein Schlüssel. Ein deutsches Ticket mit
  `returns`/`shipping`/`billing` wäre ein gemischter Prompt.
- **Die Fragen.** Eine englische Frage zu einem deutschen Ticket würde etwas
  messen, das so niemand ausliefert.

Und weil die Namen mit der Sprache wechseln, prüft das Beispiel die **Position**
der Option (`probabilities.get_index`), nicht ihren Namen , nur so lässt sich
dieselbe Erwartung an beide Sprachen stellen.

Was es *nicht* benutzt, damit der Weg sichtbar bleibt: kein async, kein Batching,
keine Permutation, keine Cache-Einstellungen. Nur die Genauigkeit kommt
vom Gerät , auf einer CPU f32, wie es `kev.serve` dort auch tut.

## Schnellstart

Ein Skript holt einen Checkpoint (mit `curl` , das `hf`-CLI ist selbst Python) und
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

Ob die Checkpoints auf **deutschen** Texten genauso zuverlässig sind, ist nicht
vorhergesagt, sondern messbar , die Checkpoints sind auf englischen Daten
veröffentlicht, die Qwen-Basis ist mehrsprachig:

```bash
cargo run --release --example deutsch -- \
    --basis ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b --vergleich
```

Ein deutsches Ticket mit allen drei Fragetypen, danach sieben Fälle, deren
Antwort nicht in Frage steht , und mit `--vergleich` dieselben Inhalte auf
englisch in der Spalte daneben. Verglichen wird über die Position der Option,
nicht über ihren Namen: Optionsnamen stehen im Prompt (`name: beschreibung`) und
sind damit Teil der Sprache. Die Zufallslinie steht unter der Tabelle, damit eine
Trefferzahl nicht besser aussieht als sie ist.

Weitere Eingabeformen: `--request anfrage.json` (genau das, was du sonst POSTen
würdest, mehrfach = Batch), `--questions fragen.json` zu einem `--state`,
`--lines` für einen State pro Zeile von stdin (Modell lädt einmal, Cache bleibt
warm), `--json` für exakt die Antwort, die der Server geschickt hätte. Diagnose
geht auf stderr, Antworten auf stdout.

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

Ein asynchroner Aufrufer nimmt `engine.system_one(&request).await` , das schiebt
den Pass vom Runtime-Thread weg, weil ein Forward-Pass CPU-gebunden ist.
`LocalEngine` ist billig zu klonen, und Klone teilen das eine geladene Modell.

## Auf deinen eigenen Tickets messen

Parität fragt, ob diese Engine der Referenz gleicht. Die andere Frage , und die,
die entscheidet, ob ein Checkpoint dir etwas nützt , ist: wie oft hat er auf
*deinen* Tickets recht, und zwar so, dass du darauf routen kannst?

```bash
cd rkev
cargo run --release --example eval -- \
    --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b \
    --questions tests/eval/questions.json --records meine-tickets.jsonl
```

Die Fragen sind die `questions`-Map einer System-One-Anfrage (oder `--request` auf
eine ganze Anfrage, die du schon hast). Die Records sind JSONL, ein Objekt pro
Zeile:

```json
{"state": "Mein Paket ist nie angekommen.", "labels": {"abteilung": "versand", "eskalation": true, "verärgerung": 1}}
```

`state` ist Text oder beliebiges JSON, wie in einer Anfrage. `labels` hält, was du
für richtig hältst: bei `choice` den Optionsnamen, bei `noul` `true`/`false`, bei
`score` den Stufenindex oder die Stufenbeschreibung. Eine Frage, die du wegläßt,
wird für den Record nicht gewertet , ein teilweise beschrifteter Satz ist also
brauchbar. Ein Label für eine Frage, die es nicht gibt, ist dagegen ein Fehler und
keine stille Null: ein Tippfehler in einer Frage-Id sähe sonst wie eine perfekte
Trefferquote auf nichts aus.

Berichtet wird pro Frage das Maß, das zum Typ passt , Trefferquote bei `choice`
samt Verwechslungen zwischen den Optionen, Trefferquote bei 0.5 plus die Trennung
beider Klassen und die AUC bei `noul`, nächstliegende Stufe plus mittlerer
absoluter Fehler bei `score`:

```text
question         type     scored  accuracy  and what else it says
abteilung        choice      120     84.2%  confidence 0.71 when right, 0.38 when wrong
eskalation       noul        120     91.7%  p(yes) 0.84 when yes, 0.11 when no, AUC 0.95
verärgerung      score        96     62.5%  mean absolute error 0.44 levels
```

Und dann der Teil, für den ein Entscheidungsmodell überhaupt da ist: **Trefferquote
nach Sicherheit.** Daraus kommt die Schwelle, über der du automatisch routen und
unter der du an einen Menschen geben kannst. `--errors <n>` zeigt die sichersten
Fehlgriffe , dort steckt meist die Formulierung einer Frage, nicht das Modell.

`scripts/local.sh --records tickets.jsonl --questions q.json` hängt denselben
Schritt an alles andere an, nach den Prüfungen und vor den Zeitmessungen.

Für CI gibt es eine Schwelle — eine niedrige Trefferquote ist nur dann ein
Fehlschlag, wenn du sagst, was niedrig heißt:

```bash
cargo run --release --example eval -- … \
    --min-accuracy 0.8 --min-accuracy verärgerung=0.6
```

`--min-accuracy 0.8` gilt für jede Frage, `id=0.6` überschreibt sie für eine — eine
Skala liegt naturgemäß unter einer dreifachen Wahl, und eine einzige Zahl für alles
wäre entweder zu lasch oder zu streng. Unterschreitet eine Frage ihre Schwelle,
endet der Lauf mit einem Fehlercode; und eine Frage **mit** Schwelle, für die nichts
beschriftet ist, gilt als unterschritten — eine Garantie ohne Belege ist keine.
`scripts/local.sh` leitet das Flag weiter.

Die Zahlen oben sind eine Formatillustration, kein gemessener Lauf. Sechs
Beispiel-Records und die passende Fragendatei liegen in `rkev/tests/eval/`,
zum Abschauen des Formats; `--batch 8` beschleunigt kurze States, `--json` gibt die
Auswertung maschinenlesbar aus, und die Batch-Größe ändert die Zahlen nicht (geprüft:
`--batch 1` und `--batch 4` liefern dasselbe JSON).

## Features

| Feature | Was es bringt | Kosten |
|---|---|---|
| `candle` (Standard) | das Modell selbst: beide Qwen-Generationen | candle 0.9, tokenizers, zip |
| `local` | Prompt, Token-Layout, Pointer-Head, `LocalEngine`; `Forward` bleibt dir | tokio (nur `spawn_blocking`) |
| , | die Wire-Format-Typen, die Fehler, die `SystemOne`-Naht | nichts |

Ohne jedes Feature (`--no-default-features`) bleibt also genau das, was ein
Aufrufer braucht, um mit etwas anderem zu reden , oder um eine Aufzeichnung zu
halten. MSRV ist 1.75 ohne `candle`, mit `candle` dessen eigener Wert.

## Beispiele

Alle in `rkev/examples/`:

| Beispiel | Wofür | Braucht |
|---|---|---|
| `decide` | Anfragen beantworten , der Server-Job im eigenen Prozess | Checkpoint |
| `sanity` | prüft die Engine gegen sich selbst und gegen eindeutige Fälle | Checkpoint |
| `measure` | f16 gegen f32, Prefix-Cache, Chunking, Batching , mit Kontrollzeilen | Checkpoint |
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
scripts/parity.sh --base <dir> --checkpoint <dir>    # gegen einen Server aufzeichnen
scripts/parity.sh --check-only --base <dir> --checkpoint <dir>   # und offline nachprüfen
```

`scripts/local.sh --help` listet den Rest. `KEV_HF` zeigt die Downloads auf einen
Spiegel (oder auf einen `file://`-Baum).

## Tests

Alle offline, keiner braucht einen Server oder Gewichte:

```bash
cd rkev
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
  Delta-Rule in Chunks von 64 Tokens. Alles exakt , die Tests verlangen
  identische Antworten.

Was offen ist:

- **Parität mit der Referenz ist nicht geprüft.** Dass die Zahlen plausibel und
  untereinander konsistent sind, heißt nicht, dass sie dieselben sind. Das ist die
  **einzige** Stelle, an der der Python-Server noch gebraucht wird, und auch dort
  genau einmal: `scripts/parity.sh` zeichnet fünf Anfragen über beide Endpunkte auf
  und vergleicht jede Wahrscheinlichkeit. Danach sind die Aufzeichnungen Dateien,
  und `--check-only` wiederholt den Vergleich offline, für immer.
- Auf einer CPU gibt es in candle kein bf16-Matmul, also wird bf16 dort mit einer
  Meldung abgelehnt statt tief in einer Projektion zu scheitern; f16 ist die
  reduzierte Präzision, die eine CPU kann.
- Vom hybriden Qwen3.5-Pfad ist die Arithmetik geprüft, aber noch **kein echter
  hybrider Checkpoint geladen** , der echte Lauf war ein attention-only Modell.
- Nichts ist quantisiert, und außer der CPU ist kein Gerät gelaufen.


### Wie Fehler laut werden

Ein falsch geladener Checkpoint rechnet weiter und antwortet plausibel , das ist
die gefährliche Sorte Fehler. Die Stellen, an denen das möglich war, weigern sich
inzwischen:

- **Ein Adapter-Tensor, den der Merge nie anfasst, verhindert das Laden.** Das war
  die leiseste Art, ein falsches Modell zu servieren: `weights.rs` sucht die
  LoRA-Gewichte unter pefts Namensschema `base_model.model.<pfad>`, und was es
  dort nicht findet, wurde einfach nicht gemergt , das Modell lief weiter, jede
  Antwort sah vernünftig aus, die Zahlen waren die eines anderen Modells. Jetzt
  muss **jeder** Tensor der Adapterdatei verbraucht sein. Betrifft er ein Gewicht,
  das das Backbone liest, ist es ein Fehler mit Namen; betrifft er ein Modul, das
  hier gar nicht läuft (einen Vokabular-Head etwa , Kevs Antworten gehen durch
  keinen), eine Warnung. `KEV_ALLOW_UNUSED=1` hebt die Weigerung auf, wenn du die
  Namen gelesen und entschieden hast.
- **Truncation und Padding des Tokenizers** sind ausdrücklich abgeschaltet, wie es
  transformers bei jedem Aufruf tut, und ein Test hält das fest (mit
  `KEV_TOKENIZER`, weil ein echter Qwen-Tokenizer hier nicht mitgeliefert werden
  kann). Ein Tokenizer, der kürzt, ist ein leise anderer Prompt.
- **Die `config.json`** wird nicht stillschweigend teilweise gelesen: Sliding-Window,
  eine andere Rotary-Variante, `rope_scaling`, Attention-Bias oder nicht aufgehende
  Kopfzahlen sind Weigerungen, keine Näherungen.
- **Die Formeln** von Prompt, Layout und Readout stehen gegen acht aus der Referenz
  portierte Testfälle, die Forward-Pässe gegen Transkriptionen von Hugging Faces
  `modeling_qwen3.py` und `modeling_qwen3_5.py`, Hidden-Unit für Hidden-Unit.

- **Die Orientierung des Pointer-Heads ist empirisch abgesichert.** Welche
  Projektion `<decide>` liest und welche jedes `</opt>`, steht nur in den Namen `q`
  und `k` , vertauscht ergibt der Head eine andere, genauso plausible Verteilung,
  und keine Form- oder Konsistenzprüfung kann das unterscheiden, weil beide Seiten
  jedes Vergleichs gleich vertauscht wären. Ein *trainierter* Head kann es:
  `examples/sanity.rs` beantwortet die eindeutigen Fälle zusätzlich mit
  `PointerHead::swapped()` und stellt beide Trefferzahlen nebeneinander. Kostet das
  Vertauschen nichts, sagt es das ausdrücklich, statt ein Ergebnis vorzutäuschen.
  Dazu wird eine `head.pt` mit mehr als zwei Projektionen abgelehnt, und ebenso
  eine mit zwei Kandidaten für denselben Namen , sonst entschiede die Reihenfolge
  in der Datei, welche Projektion welchen Zustand liest.

Was dann noch übrig bleibt und **nur** eine Aufzeichnung fangen kann: eine
numerische Abweichung ohne jedes lokale Symptom , etwa ob die Referenz an einer
Stelle rundet, wo dieser Code es nicht tut, oder ein Konfigurationsfeld anders
auslegt. Das ist ein Lesefehler in einem Port, und dagegen hilft nur, die Zahlen
einmal nebeneinanderzulegen.

Deshalb bleibt `scripts/parity.sh`, und deshalb nur einmal: danach sind die
Aufzeichnungen Dateien.
