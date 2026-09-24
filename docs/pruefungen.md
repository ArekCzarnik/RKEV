# Prüfungen

Woran dieser Port festgemacht ist, und wie ein falsch geladener Checkpoint laut
wird statt leise falsch zu antworten.

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
  Delta-Rule in Chunks von 64 Tokens. Alles exakt — die Tests verlangen
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
  hybrider Checkpoint geladen** — der echte Lauf war ein attention-only Modell.
- Nichts ist quantisiert, und außer der CPU ist kein Gerät gelaufen.


## Wie Fehler laut werden

Ein falsch geladener Checkpoint rechnet weiter und antwortet plausibel — das ist
die gefährliche Sorte Fehler. Die Stellen, an denen das möglich war, weigern sich
inzwischen:

- **Ein Adapter-Tensor, den der Merge nie anfasst, verhindert das Laden.** Das war
  die leiseste Art, ein falsches Modell zu servieren: `weights.rs` sucht die
  LoRA-Gewichte unter pefts Namensschema `base_model.model.<pfad>`, und was es
  dort nicht findet, wurde einfach nicht gemergt — das Modell lief weiter, jede
  Antwort sah vernünftig aus, die Zahlen waren die eines anderen Modells. Jetzt
  muss **jeder** Tensor der Adapterdatei verbraucht sein. Betrifft er ein Gewicht,
  das das Backbone liest, ist es ein Fehler mit Namen; betrifft er ein Modul, das
  hier gar nicht läuft (einen Vokabular-Head etwa — Kevs Antworten gehen durch
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
  und `k` — vertauscht ergibt der Head eine andere, genauso plausible Verteilung,
  und keine Form- oder Konsistenzprüfung kann das unterscheiden, weil beide Seiten
  jedes Vergleichs gleich vertauscht wären. Ein *trainierter* Head kann es:
  `examples/sanity.rs` beantwortet die eindeutigen Fälle zusätzlich mit
  `PointerHead::swapped()` und stellt beide Trefferzahlen nebeneinander. Kostet das
  Vertauschen nichts, sagt es das ausdrücklich, statt ein Ergebnis vorzutäuschen.
  Dazu wird eine `head.pt` mit mehr als zwei Projektionen abgelehnt, und ebenso
  eine mit zwei Kandidaten für denselben Namen — sonst entschiede die Reihenfolge
  in der Datei, welche Projektion welchen Zustand liest.

Was dann noch übrig bleibt und **nur** eine Aufzeichnung fangen kann: eine
numerische Abweichung ohne jedes lokale Symptom — etwa ob die Referenz an einer
Stelle rundet, wo dieser Code es nicht tut, oder ein Konfigurationsfeld anders
auslegt. Das ist ein Lesefehler in einem Port, und dagegen hilft nur, die Zahlen
einmal nebeneinanderzulegen.

Deshalb bleibt `scripts/parity.sh`, und deshalb nur einmal: danach sind die
Aufzeichnungen Dateien.

---

[← zurück zur README](../README.md)
