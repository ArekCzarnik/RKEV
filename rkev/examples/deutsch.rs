//! Kev auf deutsch: ein Ticket, drei Fragen, und die Frage dahinter — versteht
//! ein Checkpoint deutsche Texte so gut wie englische?
//!
//! Die Checkpoints sind auf englischen Daten veröffentlicht, die Basis (Qwen3
//! bzw. Qwen3.5) ist mehrsprachig. Was dabei herauskommt, ist damit nicht
//! vorhergesagt, sondern messbar: dieses Beispiel stellt dieselben Fälle einmal
//! auf deutsch und mit `--vergleich` dieselben Inhalte auf englisch, und stellt
//! die Antworten nebeneinander.
//!
//! ```bash
//! cargo run --release --example deutsch -- \
//!     --base ~/models/qwen-qwen3-0.6b-base --checkpoint ~/models/kev-0.6b
//!
//! # dieselben Inhalte auf englisch daneben
//! cargo run --release --example deutsch -- --base <base> --checkpoint <kev> --vergleich
//! ```
//!
//! Die Optionsnamen stehen im Prompt (`name: beschreibung`), sind also Teil der
//! Sprache und nicht bloß Schlüssel. Deshalb hat die deutsche Fassung deutsche
//! Namen, und verglichen wird über die Position, nicht über den Namen.

use std::path::PathBuf;
use std::process::ExitCode;

use rkev::{
    option_isolation, pointer_head, Answer, Backend, Choice, LocalEngine, Noul, Score,
    SystemOneRequest,
};

fn main() -> ExitCode {
    let optionen = match einlesen() {
        Ok(optionen) => optionen,
        Err(meldung) => {
            eprintln!("{meldung}\n\n{HILFE}");
            return ExitCode::FAILURE;
        }
    };
    match laufen(&optionen) {
        Ok(()) => ExitCode::SUCCESS,
        Err(fehler) => {
            eprintln!("{fehler}");
            ExitCode::FAILURE
        }
    }
}

struct Optionen {
    basis: PathBuf,
    checkpoint: Option<PathBuf>,
    kopf: Option<PathBuf>,
    vergleich: bool,
}

/// Welche Sprache ein Fall gestellt wird. Betrifft den State *und* die Fragen:
/// eine deutsche Frage zu einem deutschen Ticket ist der ehrliche Vergleich.
#[derive(Clone, Copy, PartialEq)]
enum Sprache {
    Deutsch,
    Englisch,
}

/// Die vier Fragen, die dieses Beispiel stellt.
#[derive(Clone, Copy)]
enum Frage {
    Abteilung,
    Eskalation,
    Verärgerung,
    Dringlichkeit,
}

/// Was bei dem Fall herauskommen muss. Bei einer Wahl die *Position* der
/// richtigen Option, weil die Namen sich mit der Sprache ändern.
#[derive(Clone, Copy)]
enum Erwartung {
    Wahl(usize),
    Ja(bool),
    Über(f64),
    Unter(f64),
}

struct Fall {
    deutsch: &'static str,
    englisch: &'static str,
    frage: Frage,
    erwartet: Erwartung,
}

fn fälle() -> Vec<Fall> {
    vec![
        Fall {
            deutsch: "Mein Paket ist nie angekommen, die Sendungsverfolgung steht seit acht Tagen.",
            englisch: "My parcel never arrived, the tracking has not moved in eight days.",
            frage: Frage::Abteilung,
            erwartet: Erwartung::Wahl(1),
        },
        Fall {
            deutsch: "Ich wurde für Bestellung 4411 zweimal abgebucht. Bitte eine Buchung zurück.",
            englisch: "I was charged twice for order 4411. Please refund one of them.",
            frage: Frage::Abteilung,
            erwartet: Erwartung::Wahl(2),
        },
        Fall {
            deutsch: "Der Pullover ist zwei Nummern zu klein. Kann ich ihn umtauschen?",
            englisch: "The jumper is two sizes too small. Can I exchange it?",
            frage: Frage::Abteilung,
            erwartet: Erwartung::Wahl(0),
        },
        Fall {
            deutsch:
                "Zum dritten Mal schreibe ich. Niemand antwortet. Ich will mein Geld zurück, heute.",
            englisch: "Third time writing. Nobody answers. I want my money back today.",
            frage: Frage::Eskalation,
            erwartet: Erwartung::Ja(true),
        },
        Fall {
            deutsch: "Wollte nur kurz danke sagen, die Schuhe sind wunderschön.",
            englisch: "Just wanted to say thank you, the shoes are lovely.",
            frage: Frage::Eskalation,
            erwartet: Erwartung::Ja(false),
        },
        Fall {
            deutsch: "Ich bin absolut wütend über diesen Service.",
            englisch: "I am absolutely furious about this service.",
            frage: Frage::Verärgerung,
            erwartet: Erwartung::Über(1.4),
        },
        Fall {
            deutsch: "Keine Eile, wann es Ihnen gerade passt.",
            englisch: "No rush at all, whenever it suits you.",
            frage: Frage::Dringlichkeit,
            erwartet: Erwartung::Unter(0.6),
        },
    ]
}

/// Eine Anfrage mit genau einer Frage, in der gewählten Sprache.
fn anfrage(text: &str, frage: Frage, sprache: Sprache) -> SystemOneRequest {
    let deutsch = sprache == Sprache::Deutsch;
    let anfrage = SystemOneRequest::new(text.to_string());
    match frage {
        Frage::Abteilung if deutsch => anfrage.ask(
            "abteilung",
            Choice::new("Welches Team soll das übernehmen?")
                .option(
                    "rückgabe",
                    "Umtausch, Rückgabe, falsche oder beschädigte Ware",
                )
                .option("versand", "Lieferstatus, Verzug, verlorene Pakete")
                .option("zahlung", "Abbuchungen, Rechnungen, Zahlungsprobleme"),
        ),
        Frage::Abteilung => anfrage.ask(
            "abteilung",
            Choice::new("Which team should handle this?")
                .option("returns", "Exchanges, refunds, wrong or damaged items")
                .option("shipping", "Delivery status, delays, lost packages")
                .option("billing", "Charges, invoices, payment problems"),
        ),
        Frage::Eskalation if deutsch => anfrage.ask(
            "eskalation",
            Noul::new("Muss sich dringend ein Mensch darum kümmern?"),
        ),
        Frage::Eskalation => anfrage.ask(
            "eskalation",
            Noul::new("Does this need urgent human attention?"),
        ),
        Frage::Verärgerung if deutsch => anfrage.ask(
            "verärgerung",
            Score::new("Wie verärgert ist die Kundin oder der Kunde?")
                .level("ruhig")
                .level("verärgert")
                .level("sehr wütend"),
        ),
        Frage::Verärgerung => anfrage.ask(
            "verärgerung",
            Score::new("How frustrated is the customer?")
                .level("calm")
                .level("frustrated")
                .level("very angry"),
        ),
        Frage::Dringlichkeit if deutsch => anfrage.ask(
            "dringlichkeit",
            Score::new("Wie dringend ist das Ticket?")
                .level("kann warten")
                .level("diese Woche")
                .level("heute"),
        ),
        Frage::Dringlichkeit => anfrage.ask(
            "dringlichkeit",
            Score::new("How urgent is this ticket?")
                .level("can wait")
                .level("this week")
                .level("today"),
        ),
    }
}

/// Das ausgearbeitete Beispiel: ein Ticket, alle drei Fragetypen auf einmal.
fn ticket() -> SystemOneRequest {
    SystemOneRequest::new(
        "Die Schuhe kamen zwei Wochen zu spät und in der falschen Größe. \
         Außerdem sehe ich zwei Abbuchungen auf meiner Karte.",
    )
    .ask(
        "abteilung",
        Choice::new("Welches Team soll das übernehmen?")
            .option(
                "rückgabe",
                "Umtausch, Rückgabe, falsche oder beschädigte Ware",
            )
            .option("versand", "Lieferstatus, Verzug, verlorene Pakete")
            .option("zahlung", "Abbuchungen, Rechnungen, Zahlungsprobleme"),
    )
    .ask(
        "eskalation",
        Noul::new("Muss sich dringend ein Mensch darum kümmern?"),
    )
    .ask(
        "verärgerung",
        Score::new("Wie verärgert ist die Kundin oder der Kunde?")
            .level("ruhig")
            .level("verärgert")
            .level("sehr wütend"),
    )
}

fn laufen(optionen: &Optionen) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = optionen.checkpoint.as_deref();
    let kopf = optionen.kopf.clone().unwrap_or_else(|| {
        let ordner = checkpoint.unwrap_or(&optionen.basis);
        [ordner.join("head.pt"), ordner.join("head.safetensors")]
            .into_iter()
            .find(|pfad| pfad.exists())
            .unwrap_or_else(|| ordner.join("head.pt"))
    });

    let hintergrund = Backend::open(&optionen.basis, checkpoint)?;
    let hybrid = hintergrund.is_hybrid();
    let genauigkeit = hintergrund.dtype();
    let mut motor = LocalEngine::new(hintergrund, pointer_head(&kopf)?);
    if let Some(isolation) = option_isolation(&kopf)? {
        motor = motor.with_option_isolation(isolation);
    }

    println!(
        "{} in {genauigkeit:?}\n",
        if hybrid {
            "hybride Basis (Attention und Gated DeltaNet)"
        } else {
            "Basis nur mit Attention"
        }
    );

    // --- das ausgearbeitete Beispiel ---
    let beispiel = ticket();
    let antwort = motor.system_one_blocking(&beispiel)?;
    println!("Ticket: {}\n", beispiel.state.as_str().unwrap_or_default());
    for (id, einzeln) in &antwort.answers {
        println!("{}", beschreiben(id, einzeln));
    }
    println!(
        "\n{} Tokens im Prompt, {:.0} ms",
        antwort.usage.input_tokens,
        antwort.latency_ms.unwrap_or(f64::NAN)
    );

    // --- versteht der Checkpoint deutsch? ---
    println!("\nFälle, deren Antwort nicht in Frage steht:\n");
    println!(
        "{:<70} {:>22}{}",
        "Ticket",
        "deutsch",
        if optionen.vergleich {
            format!("{:>22}", "englisch")
        } else {
            String::new()
        }
    );

    let fälle = fälle();
    let mut treffer_de = 0;
    let mut treffer_en = 0;
    for fall in &fälle {
        let de = motor.system_one_blocking(&anfrage(fall.deutsch, fall.frage, Sprache::Deutsch))?;
        let (text_de, ok_de) = urteil(erste(&de)?, &fall.erwartet);
        treffer_de += usize::from(ok_de);

        let mut zeile = format!(
            "{:<70} {:>22}",
            kürzen(fall.deutsch, 68),
            format!("{text_de}{}", if ok_de { "" } else { " ✗" })
        );
        if optionen.vergleich {
            let en = motor.system_one_blocking(&anfrage(
                fall.englisch,
                fall.frage,
                Sprache::Englisch,
            ))?;
            let (text_en, ok_en) = urteil(erste(&en)?, &fall.erwartet);
            treffer_en += usize::from(ok_en);
            zeile.push_str(&format!(
                "{:>22}",
                format!("{text_en}{}", if ok_en { "" } else { " ✗" })
            ));
        }
        println!("{zeile}");
    }

    println!("\n{treffer_de} von {} auf deutsch", fälle.len());
    if optionen.vergleich {
        println!("{treffer_en} von {} auf englisch", fälle.len());
        println!(
            "\nDer Unterschied zwischen den beiden Zahlen ist das Ergebnis, nicht die\n\
             Trefferzahl allein: die Checkpoints sind auf englischen Daten\n\
             veröffentlicht, die Basis ist mehrsprachig. Bei Zufall wären es etwa\n\
             {:.1} von {}.",
            zufall(&fälle),
            fälle.len()
        );
    } else {
        println!(
            "Bei Zufall wären es etwa {:.1}. Mit --vergleich stehen dieselben \
             Inhalte auf englisch daneben.",
            zufall(&fälle)
        );
    }
    Ok(())
}

/// Wie viele Treffer reines Raten im Schnitt gäbe: 1/3 bei den Wahlfragen,
/// 1/2 bei ja/nein, und bei den Skalen liegt eine von drei Stufen richtig.
fn zufall(fälle: &[Fall]) -> f64 {
    fälle
        .iter()
        .map(|fall| match fall.frage {
            Frage::Abteilung => 1.0 / 3.0,
            Frage::Eskalation => 0.5,
            Frage::Verärgerung | Frage::Dringlichkeit => 1.0 / 3.0,
        })
        .sum()
}

fn erste(antwort: &rkev::SystemOneResponse) -> Result<&Answer, &'static str> {
    antwort
        .answers
        .values()
        .next()
        .ok_or("die Engine hat nichts geantwortet")
}

fn urteil(antwort: &Answer, erwartet: &Erwartung) -> (String, bool) {
    let richtig = match (antwort, erwartet) {
        (
            Answer::Choice {
                probabilities,
                choice,
                ..
            },
            Erwartung::Wahl(stelle),
        ) => probabilities
            .get_index(*stelle)
            .map(|(name, _)| name == choice)
            .unwrap_or(false),
        (Answer::Noul { noul }, Erwartung::Ja(ja)) => (*noul > 0.5) == *ja,
        (Answer::Score { score, .. }, Erwartung::Über(grenze)) => score > grenze,
        (Answer::Score { score, .. }, Erwartung::Unter(grenze)) => score < grenze,
        _ => false,
    };
    (kurz(antwort), richtig)
}

/// Eine Antwort in einer Zeile, für die Tabelle.
fn kurz(antwort: &Answer) -> String {
    match antwort {
        Answer::Noul { noul } => format!("{} {noul:.2}", if *noul > 0.5 { "ja" } else { "nein" }),
        Answer::Choice {
            choice,
            probabilities,
            ..
        } => format!(
            "{choice} {:.2}",
            probabilities.get(choice).copied().unwrap_or(f64::NAN)
        ),
        Answer::Score { score, .. } => format!("Stufe {score:.2}"),
    }
}

/// Eine Antwort ausführlich, mit der ganzen Verteilung darunter — darum geht es
/// bei Kev: die Verteilung ist die Antwort, nicht das Etikett.
fn beschreiben(id: &str, antwort: &Answer) -> String {
    match antwort {
        Answer::Noul { noul } => format!(
            "{id:<14} {:<14} {noul:.2}",
            if *noul > 0.5 { "ja" } else { "nein" }
        ),
        Answer::Choice {
            choice,
            confidence,
            probabilities,
        } => {
            let mut zeilen = vec![format!(
                "{id:<14} {choice:<14} {:.2}   Sicherheit {confidence:.2}",
                probabilities.get(choice).copied().unwrap_or(f64::NAN)
            )];
            let mut rest: Vec<_> = probabilities
                .iter()
                .filter(|(name, _)| *name != choice)
                .collect();
            rest.sort_by(|a, b| b.1.total_cmp(a.1));
            for (name, wert) in rest {
                zeilen.push(format!("{:<14} {name:<14} {wert:.2}", ""));
            }
            zeilen.join("\n")
        }
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities,
        } => {
            let nächste = legend
                .get(&format!("{}", score.round() as i64))
                .map(String::as_str)
                .unwrap_or("");
            let mut zeilen = vec![format!(
                "{id:<14} {nächste:<14} Stufe {score:.2}   Sicherheit {confidence:.2}"
            )];
            for (stelle, wert) in probabilities {
                let stufe = legend.get(stelle).map(String::as_str).unwrap_or(stelle);
                zeilen.push(format!("{:<14} {stufe:<14} {wert:.2}", ""));
            }
            zeilen.join("\n")
        }
    }
}

fn kürzen(text: &str, breite: usize) -> String {
    if text.chars().count() <= breite {
        return text.to_string();
    }
    format!("{}…", text.chars().take(breite - 1).collect::<String>())
}

const HILFE: &str = "\
Aufruf: deutsch --basis <ordner> [--checkpoint <ordner>] [--kopf <datei>] [--vergleich]

  --basis       das Basismodell: config.json und die safetensors dazu
  --checkpoint  der Kev-Checkpoint: Adapter, head.pt, Tokenizer
  --kopf        der Pointer-Head, falls nicht <checkpoint>/head.pt
  --vergleich   dieselben Inhalte zusätzlich auf englisch

Die englischen Namen --base, --head und --compare gehen auch, damit die
Aufrufe zu den anderen Beispielen passen.";

fn einlesen() -> Result<Optionen, String> {
    let mut basis = None;
    let mut checkpoint = None;
    let mut kopf = None;
    let mut vergleich = false;

    let mut argumente = std::env::args().skip(1);
    while let Some(argument) = argumente.next() {
        let mut wert = || {
            argumente
                .next()
                .ok_or_else(|| format!("{argument} braucht einen Wert"))
        };
        match argument.as_str() {
            "--basis" | "--base" => basis = Some(PathBuf::from(wert()?)),
            "--checkpoint" => checkpoint = Some(PathBuf::from(wert()?)),
            "--kopf" | "--head" => kopf = Some(PathBuf::from(wert()?)),
            "--vergleich" | "--compare" => vergleich = true,
            "-h" | "--help" | "--hilfe" => return Err(String::from("deutsch")),
            sonst => return Err(format!("unbekanntes Argument {sonst}")),
        }
    }
    Ok(Optionen {
        basis: basis.ok_or("--basis wird gebraucht")?,
        checkpoint,
        kopf,
        vergleich,
    })
}
