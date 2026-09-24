//! The request and response shapes are checked against the worked example in
//! the Kev README, so a refactor cannot silently change what goes on the wire.

use rkev::{Answer, Choice, Noul, Score, SystemOneRequest, SystemOneResponse};
use serde_json::json;

fn readme_request() -> SystemOneRequest {
    SystemOneRequest::new(
        "Shoes arrived two weeks late and in the wrong size. \
         Also I see two charges on my card.",
    )
    .model("kev-latest")
    .ask(
        "department",
        Choice::new("Which team should handle this?")
            .option("returns", "Exchanges, refunds, wrong or damaged items")
            .option("shipping", "Delivery status, delays, lost packages")
            .option("billing", "Charges, invoices, payment problems"),
    )
    .ask(
        "escalate",
        Noul::new("Does this need urgent human attention?"),
    )
    .ask(
        "frustration",
        Score::new("How frustrated is the customer?")
            .level("Calm")
            .level("Frustrated")
            .level("Very angry"),
    )
}

#[test]
fn request_matches_the_readme_example() {
    let sent = serde_json::to_value(readme_request()).unwrap();

    assert_eq!(
        sent,
        json!({
            "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
            "model": "kev-latest",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "returns": "Exchanges, refunds, wrong or damaged items",
                        "shipping": "Delivery status, delays, lost packages",
                        "billing": "Charges, invoices, payment problems"
                    }
                },
                "escalate": {
                    "type": "noul",
                    "instructions": "Does this need urgent human attention?"
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]
                }
            }
        })
    );
}

#[test]
fn choice_options_keep_the_order_they_were_added_in() {
    // Option order is part of the request and can move the answer, so it has
    // to survive serialisation.
    let question = Choice::new("Which team?")
        .option("shipping", "Delivery")
        .option("returns", "Refunds")
        .option_bare("billing");

    let sent = serde_json::to_string(&question).unwrap();

    assert_eq!(
        sent,
        r#"{"instructions":"Which team?","criteria":{"shipping":"Delivery","returns":"Refunds","billing":null}}"#
    );
}

#[test]
fn noul_criteria_are_sent_as_true_and_false_keys() {
    let question = Noul::new("Is this urgent?")
        .yes("Needs a human today")
        .no("Can wait for the queue");

    let sent = serde_json::to_value(rkev::Question::from(question)).unwrap();

    assert_eq!(sent["criteria"]["true"], json!("Needs a human today"));
    assert_eq!(sent["criteria"]["false"], json!("Can wait for the queue"));
}

#[test]
fn a_question_without_criteria_omits_the_field() {
    let sent = serde_json::to_value(rkev::Question::from(Noul::new("Urgent?"))).unwrap();

    assert_eq!(sent, json!({"type": "noul", "instructions": "Urgent?"}));
}

#[test]
fn the_state_can_be_a_structured_document() {
    let request = SystemOneRequest::new(json!({
        "subject": "Wrong size",
        "body": "Shoes arrived late."
    }));

    let sent = serde_json::to_value(request).unwrap();

    assert_eq!(sent["state"]["subject"], json!("Wrong size"));
    // No model until the client fills one in.
    assert!(sent.get("model").is_none());
}

const README_RESPONSE: &str = r#"{
  "model": "kev-latest",
  "answers": {
    "department":  { "type": "choice", "choice": "returns", "confidence": 0.21,
                     "probabilities": { "returns": 0.47, "shipping": 0.28, "billing": 0.25 } },
    "escalate":    { "type": "noul", "noul": 0.93 },
    "frustration": { "type": "score", "score": 1.44, "confidence": 0.78,
                     "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
                     "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 } }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },
  "latency_ms": 495
}"#;

#[test]
fn response_from_the_readme_decodes_into_typed_answers() {
    let response: SystemOneResponse = serde_json::from_str(README_RESPONSE).unwrap();

    assert_eq!(response.model, "kev-latest");
    assert_eq!(response.usage.input_tokens, 101);
    assert_eq!(response.latency_ms, Some(495.0));

    assert_eq!(
        response.answer("department").unwrap().as_choice(),
        Some("returns")
    );
    assert_eq!(response.answer("escalate").unwrap().as_noul(), Some(0.93));
    assert_eq!(
        response.answer("frustration").unwrap().as_score(),
        Some(1.44)
    );
}

#[test]
fn a_noul_answer_reports_no_confidence_or_distribution() {
    let response: SystemOneResponse = serde_json::from_str(README_RESPONSE).unwrap();
    let escalate = response.answer("escalate").unwrap();

    // The probability of yes is the whole answer; there is nothing else to report.
    assert_eq!(escalate.confidence(), None);
    assert!(escalate.probabilities().is_none());
    assert_eq!(escalate.as_choice(), None);
}

#[test]
fn top_returns_the_most_likely_label_of_a_distribution() {
    let response: SystemOneResponse = serde_json::from_str(README_RESPONSE).unwrap();

    assert_eq!(
        response.answer("department").unwrap().top(),
        Some(("returns", 0.47))
    );
    // For a score the label is the level index; the legend names it.
    let frustration = response.answer("frustration").unwrap();
    assert_eq!(frustration.top(), Some(("1", 0.56)));
    assert_eq!(
        frustration.legend().unwrap().get("1").map(String::as_str),
        Some("Frustrated")
    );
}

#[test]
fn an_unknown_question_id_is_absent_rather_than_a_panic() {
    let response: SystemOneResponse = serde_json::from_str(README_RESPONSE).unwrap();

    assert!(response.answer("no_such_question").is_none());
}

#[test]
fn optional_response_fields_may_be_missing() {
    // Only `model` and `answers` are guaranteed; usage and latency are not.
    let response: SystemOneResponse = serde_json::from_str(
        r#"{"model":"kev-latest","answers":{"a":{"type":"noul","noul":0.5}}}"#,
    )
    .unwrap();

    assert_eq!(response.usage.output_tokens, 0);
    assert_eq!(response.latency_ms, None);
    assert!(matches!(
        response.answer("a").unwrap(),
        Answer::Noul { noul } if *noul == 0.5
    ));
}

#[test]
fn a_request_survives_a_round_trip_through_json() {
    // The request types deserialise as well as serialise, so a recorded request
    // can be replayed - which is what examples/parity.rs does with the body the
    // server was sent.
    let original = readme_request();
    let json = serde_json::to_string(&original).unwrap();

    let parsed: SystemOneRequest = serde_json::from_str(&json).unwrap();

    assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
    assert_eq!(
        parsed.questions.keys().collect::<Vec<_>>(),
        original.questions.keys().collect::<Vec<_>>(),
        "question order is part of the request"
    );
}
