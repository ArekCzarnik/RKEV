# Auf eigenen Tickets messen

Nicht ob die Engine der Referenz gleicht, sondern ob der Checkpoint deine Tickets
richtig entscheidet — mit Kalibrierung und einer Schwelle für CI.

Parität fragt, ob diese Engine der Referenz gleicht. Die andere Frage — und die,
die entscheidet, ob ein Checkpoint dir etwas nützt — ist: wie oft hat er auf
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
wird für den Record nicht gewertet — ein teilweise beschrifteter Satz ist also
brauchbar. Ein Label für eine Frage, die es nicht gibt, ist dagegen ein Fehler und
keine stille Null: ein Tippfehler in einer Frage-Id sähe sonst wie eine perfekte
Trefferquote auf nichts aus.

Berichtet wird pro Frage das Maß, das zum Typ passt — Trefferquote bei `choice`
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
Fehlgriffe — dort steckt meist die Formulierung einer Frage, nicht das Modell.

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

## Auf deutsch

Ob die Checkpoints auf **deutschen** Texten genauso zuverlässig sind, ist nicht
vorhergesagt, sondern messbar — die Checkpoints sind auf englischen Daten
veröffentlicht, die Qwen-Basis ist mehrsprachig:

```bash
cargo run --release --example deutsch -- \
    --basis ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b --vergleich
```

Ein deutsches Ticket mit allen drei Fragetypen, danach sieben Fälle, deren
Antwort nicht in Frage steht — und mit `--vergleich` dieselben Inhalte auf
englisch in der Spalte daneben. Verglichen wird über die Position der Option,
nicht über ihren Namen: Optionsnamen stehen im Prompt (`name: beschreibung`) und
sind damit Teil der Sprache. Die Zufallslinie steht unter der Tabelle, damit eine
Trefferzahl nicht besser aussieht als sie ist.

Weitere Eingabeformen: `--request anfrage.json` (genau das, was du sonst POSTen
würdest, mehrfach = Batch), `--questions fragen.json` zu einem `--state`,
`--lines` für einen State pro Zeile von stdin (Modell lädt einmal, Cache bleibt
warm), `--json` für exakt die Antwort, die der Server geschickt hätte. Diagnose
geht auf stderr, Antworten auf stdout.

---

[← zurück zur README](../README.md)
