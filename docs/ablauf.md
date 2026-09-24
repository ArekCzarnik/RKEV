# Vom Ticket zur Antwort

Der Weg einer Anfrage durch RKev, in vier Schritten mit dem Modul zu jedem —
und was das deutsche Beispiel davon benutzt.

Vier Schritte, vier Module. Jeder Schritt ist im Code an einer Stelle, und die
Reihenfolge ist immer dieselbe — es gibt keine Schleife und keine Generierung.

## 1. Anfrage → Text (`rkev/src/prompt.rs`)

Der State wird zu Text. Ein String bleibt, wie er ist; ein JSON-Objekt wird zu
`schlüssel: wert` pro Zeile, eine Liste zu `- eintrag`, verschachteltes eingerückt.
Bei Skalaren zählt die Schreibweise: `True` und `False` statt `true`/`false`, und
`null` wird zu **nichts** — also `schlüssel: ` mit leerem Wert. Das ist kein
Geschmack, sondern Bedingung: die Referenz-Implementierung (siehe [Herkunft](../README.md#herkunft)) baut
genau diesen Text, und ein anderer Text ist eine andere Antwort.
`tests/upstream_unit.rs` hält sechs dieser Ausgaben an deren eigenen Testfällen
fest.

Jede Frage wird zu Anweisungen plus einer Liste von **Optionstexten**, jeweils
`name: beschreibung`. Die drei Fragetypen unterscheiden sich nur darin, was die
Optionen sind:

| Typ | Optionen im Prompt | Schlüssel in der Antwort |
|---|---|---|
| `noul` | zwei, `no` und `yes` (plus deren Beschreibungen) | `false`, `true` — die Antwort ist die Wahrscheinlichkeit der zweiten |
| `choice` | deine Optionen, in deiner Reihenfolge | die Optionsnamen |
| `score` | die Stufenbeschreibungen, niedrigste zuerst | die Stufenindizes `0`, `1`, … |

## 2. Text → Tokens (`rkev/src/encode.rs`)

Fünf selten benutzte Qwen-Spezialtokens dienen als Trennzeichen, damit keine
Embedding-Zeilen dazukommen müssen — der LoRA-Adapter gibt ihnen ihre Bedeutung:

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
  kausale Maske — nur so können die Fragen einander nicht lesen.
- **Die Positionen.** Jeder Zweig fängt direkt hinter dem State wieder an. Der
  State wird also einmal verarbeitet, egal wie viele Fragen mitkommen.
- **Die Ableseplätze.** Pro Frage der Index ihres `<decide>` und die Indizes
  aller `</opt>`.

Außerdem wird Aufrufertext maskiert: `<|name|>` wird zu `<¦name¦>`. Ohne das
könnte ein State eine Frage aufmachen, denn ein Tokenizer erkennt seine
Spezialtokens auch mitten im Text. Der State wird auf 8192 Tokens gekürzt
(trainiert wurde auf 384 State-Tokens und 1024 für State plus einen Zweig).

## 3. Tokens → Hidden States (`src/backend.rs`, `src/qwen3.rs`, `src/qwen3_5.rs`)

Hier fängt Qwen an. `Backend` setzt drei Dinge zusammen:

1. den **Tokenizer** des Checkpoints (Truncation und Padding ausdrücklich aus,
   wie es transformers bei jedem Aufruf macht),
2. die **Basisgewichte** aus den safetensors, mit dem **LoRA-Adapter des
   Checkpoints hineingerechnet** — in f32, *bevor* in die Rechengenauigkeit
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
  der `config.json` — bei den veröffentlichten Basen sind es drei von vier. Und
  eine Rekurrenz kann man nicht bitten, die Tokens einer anderen Frage zu
  überspringen: sie läuft die Tokens ab. Deshalb wird pro Frage eine eigene
  kausale Zeile gerechnet — State, dann ihr Zweig. Das ist exakt statt maskiert
  und genau das, was die Referenz dort auch tut.

Im Layout steht der State ohnehin nur einmal. Als **eigener** Pass, von dem die
Fragen dann fortsetzen, läuft er, wenn es sich lohnt: auf einer rekurrenten Basis
immer (sonst würde jede Frage den ganzen State erneut ablaufen), auf einer
attention-only ab 384 State-Tokens — darunter spart es nichts, weil der gepackte
Pass den State schon einmal rechnet. Fortgesetzt wird bei Attention über die
gespeicherten Keys und Values, bei der Rekurrenz über Zustandsmatrix und
Faltungsfenster. Die letzten vier Zustände bleiben über Anfragen hinweg im Cache,
nach ihren Tokens geschlüsselt, wie es `kev.serve` auch macht.

## 4. Hidden States → Antwort (`rkev/src/readout.rs`)

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

- `noul`: `p(yes)`, und keine Confidence — die Wahrscheinlichkeit *ist* die
  Antwort.
- `choice`: die wahrscheinlichste Option, Confidence `(p_max − 1/K)/(1 − 1/K)`.
- `score`: der mittlere Stufenindex, Confidence `1 − E|Stufe − Modus|/(L−1)`.

Alles auf vier Dezimalen gerundet, wie in der Referenz. Die Confidence ist ein Maß
für die *Form* der Verteilung, keine gemessene Trefferquote — TypeSafes
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

## Was das deutsche Beispiel davon benutzt

`rkev/examples/deutsch.rs` geht genau diesen Weg, einmal pro Anfrage:

| Im Beispiel | Was dahinter passiert |
|---|---|
| `Backend::open(&basis, checkpoint)` | Tokenizer laden, Basis + LoRA mergen, Backbone aus der `config.json` wählen (Schritt 3) |
| `pointer_head(&kopf)` | `head.pt` lesen: die zwei Projektionen **und** die Temperatur (Schritt 4) |
| `option_isolation(&kopf)` | fragt den Checkpoint, für welches Layout er trainiert wurde; das falsche zu servieren wäre ein leise anderer Prompt |
| `LocalEngine::new(…)` | verbindet beide Hälften; `Clone` ist billig, das Modell wird nicht zweimal geladen |
| `SystemOneRequest::new(text)` | der State — hier ein deutscher String, also geht er unverändert in Schritt 1 |
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
der Option (`probabilities.get_index`), nicht ihren Namen — nur so lässt sich
dieselbe Erwartung an beide Sprachen stellen.

Was es *nicht* benutzt, damit der Weg sichtbar bleibt: kein async, kein Batching,
keine Permutation, keine Cache-Einstellungen. Nur die Genauigkeit kommt
vom Gerät — auf einer CPU f32, wie es `kev.serve` dort auch tut.

---

[← zurück zur README](../README.md)
