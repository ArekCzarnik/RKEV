//! The local engine end to end, over a backend with no model in it.
//!
//! The stub returns hidden states chosen so that the pointer head produces
//! whichever distribution a test asks for. That fixes the one thing a test
//! cannot check here — what a real backbone would say — and leaves everything
//! Kev-specific under test: the prompt, the token layout, the question
//! isolation, the readout, and the answers.
//!
//! The worked example is the one in the Kev README, and the numbers asserted
//! here are the ones it publishes.

#![cfg(feature = "local")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use kev_client::{
    Choice, Error, Forward, Limits, Linear, LocalEngine, Noul, Pass, PointerHead, Result, Score,
    SystemOne, SystemOneRequest, DECIDE, OPTION, OPTION_END, QUESTION, STATE,
};

/// The ids this stub's tokenizer gives Kev's five delimiters. Caller text
/// tokenises one byte per token, offset well clear of these.
const STATE_ID: u32 = 1;
const QUESTION_ID: u32 = 2;
const OPTION_ID: u32 = 3;
const OPTION_END_ID: u32 = 4;
const DECIDE_ID: u32 = 5;
const TEXT_BASE: u32 = 1000;

/// What the engine asked the backend to do.
#[derive(Default)]
struct Log {
    /// Every piece of text tokenised, in order.
    texts: Vec<String>,
    passes: Vec<Recorded>,
}

struct Recorded {
    ids: Vec<u32>,
    positions: Vec<u32>,
    segments: Vec<u32>,
    readout: Vec<usize>,
}

impl Recorded {
    fn as_pass(&self) -> Pass<'_> {
        Pass {
            ids: &self.ids,
            positions: &self.positions,
            segments: &self.segments,
            readout: &self.readout,
        }
    }

    /// The token indices belonging to question `question` (1-based).
    fn branch(&self, question: u32) -> Vec<usize> {
        (0..self.ids.len())
            .filter(|index| self.segments[*index] == question)
            .collect()
    }
}

struct Stub {
    /// One target distribution per question, in the order the questions are
    /// asked; consumed as the engine works through them.
    targets: Vec<Vec<f64>>,
    served: usize,
    /// Answer through `Pass::rows`, the way a backbone that cannot honour the
    /// block-causal mask has to.
    rows: bool,
    log: Arc<Mutex<Log>>,
    passes: Arc<AtomicUsize>,
}

impl Stub {
    fn new(targets: Vec<Vec<f64>>) -> (Self, Arc<Mutex<Log>>, Arc<AtomicUsize>) {
        let log = Arc::new(Mutex::new(Log::default()));
        let passes = Arc::new(AtomicUsize::new(0));
        let stub = Stub {
            targets,
            served: 0,
            rows: false,
            log: Arc::clone(&log),
            passes: Arc::clone(&passes),
        };
        (stub, log, passes)
    }

    /// Hidden states for one pass, without recording it: a `<decide>` position
    /// gets `[sqrt(2), 0]`, an option's `</opt>` gets `[ln p, 0]`. With the
    /// identity pointer head below that makes the head's logit for an option
    /// exactly `ln p`, so the softmax is the target distribution.
    fn states(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> {
        let mut states = Vec::with_capacity(pass.readout.len());
        let mut option = 0;
        for position in pass.readout {
            match pass.ids[*position] {
                DECIDE_ID => {
                    self.served += 1;
                    option = 0;
                    states.push(vec![2f32.sqrt(), 0.0]);
                }
                OPTION_END_ID => {
                    let target = self.targets.get(self.served - 1).ok_or_else(|| {
                        Error::Engine(format!("stub has no target for question {}", self.served))
                    })?;
                    states.push(vec![target[option].ln() as f32, 0.0]);
                    option += 1;
                }
                id => {
                    return Err(Error::Engine(format!(
                        "the readout asked for position {position}, which holds {id}, \
                         not a <decide> or </opt> token"
                    )))
                }
            }
        }
        Ok(states)
    }
}

impl Forward for Stub {
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>> {
        self.log.lock().unwrap().texts.push(text.to_string());
        Ok(text.bytes().map(|b| TEXT_BASE + u32::from(b)).collect())
    }

    fn delimiter(&mut self, token: &str) -> Result<u32> {
        match token {
            STATE => Ok(STATE_ID),
            QUESTION => Ok(QUESTION_ID),
            OPTION => Ok(OPTION_ID),
            OPTION_END => Ok(OPTION_END_ID),
            DECIDE => Ok(DECIDE_ID),
            other => Err(Error::Engine(format!("unknown delimiter {other}"))),
        }
    }

    fn hidden(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> {
        self.passes.fetch_add(1, Ordering::SeqCst);
        self.log.lock().unwrap().passes.push(Recorded {
            ids: pass.ids.to_vec(),
            positions: pass.positions.to_vec(),
            segments: pass.segments.to_vec(),
            readout: pass.readout.to_vec(),
        });

        if !self.rows {
            return self.states(pass);
        }
        // One independent causal row per question, concatenated in order -
        // what a Gated DeltaNet backbone is left with.
        let mut states = Vec::new();
        for row in pass.rows() {
            states.extend(self.states(&row.as_pass())?);
        }
        Ok(states)
    }
}

/// The identity pointer head: `logit(option) = h_opt . h_decide / sqrt(2)`.
/// A real one carries a checkpoint's trained projections.
fn identity_head() -> PointerHead {
    let projection = || Linear::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2).unwrap();
    PointerHead::new(projection(), projection()).unwrap()
}

/// The support ticket from the Kev README, and the distributions it reports.
fn readme_request() -> SystemOneRequest {
    SystemOneRequest::new(
        "Shoes arrived two weeks late and in the wrong size. \
         Also I see two charges on my card.",
    )
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

/// The README's probabilities. `frustration`'s lowest level is reported as
/// `0.00` there, which is a rounded number, not a zero.
fn readme_targets() -> Vec<Vec<f64>> {
    vec![
        vec![0.47, 0.28, 0.25],
        vec![0.07, 0.93],
        vec![0.00004, 0.55996, 0.44],
    ]
}

fn engine(targets: Vec<Vec<f64>>) -> (LocalEngine, Arc<Mutex<Log>>, Arc<AtomicUsize>) {
    let (stub, log, passes) = Stub::new(targets);
    (LocalEngine::new(stub, identity_head()), log, passes)
}

#[test]
fn the_answers_reproduce_the_worked_example_from_the_kev_readme() {
    let (engine, _, _) = engine(readme_targets());

    let response = engine.system_one_blocking(&readme_request()).unwrap();

    let department = response.answer("department").unwrap();
    assert_eq!(department.as_choice(), Some("returns"));
    assert_eq!(department.probabilities().unwrap()["returns"], 0.47);
    assert_eq!(department.probabilities().unwrap()["shipping"], 0.28);
    assert_eq!(department.probabilities().unwrap()["billing"], 0.25);
    // (p_max - 1/K) / (1 - 1/K), which the README prints as 0.21.
    assert_eq!(department.confidence(), Some(0.205));

    assert_eq!(response.answer("escalate").unwrap().as_noul(), Some(0.93));

    let frustration = response.answer("frustration").unwrap();
    // The mean level index, and 1 - E|level - mode| / (L - 1).
    assert_eq!(frustration.as_score(), Some(1.44));
    assert_eq!(frustration.confidence(), Some(0.78));
    assert_eq!(frustration.probabilities().unwrap()["0"], 0.0);
    assert_eq!(frustration.probabilities().unwrap()["1"], 0.56);
    assert_eq!(frustration.legend().unwrap()["2"], "Very angry");
    assert_eq!(frustration.top(), Some(("1", 0.56)));
}

#[test]
fn the_answers_come_back_in_the_order_the_questions_were_asked() {
    let (engine, _, _) = engine(readme_targets());

    let response = engine.system_one_blocking(&readme_request()).unwrap();

    let ids: Vec<&str> = response.answers.keys().map(String::as_str).collect();
    assert_eq!(ids, ["department", "escalate", "frustration"]);
}

#[test]
fn the_prompt_is_the_state_then_each_question_with_its_options() {
    let (engine, log, _) = engine(readme_targets());

    engine.system_one_blocking(&readme_request()).unwrap();

    let texts = &log.lock().unwrap().texts;
    assert_eq!(
        texts[..12],
        [
            "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
            "Which team should handle this?",
            "returns: Exchanges, refunds, wrong or damaged items",
            "shipping: Delivery status, delays, lost packages",
            "billing: Charges, invoices, payment problems",
            "Does this need urgent human attention?",
            // A yes/no question is two options; its criteria would describe them.
            "no",
            "yes",
            "How frustrated is the customer?",
            "Calm",
            "Frustrated",
            "Very angry",
        ]
    );
}

#[test]
fn a_structured_state_is_rendered_as_labelled_text() {
    let (engine, log, _) = engine(vec![vec![0.5, 0.5]]);
    let request = SystemOneRequest::from_state(&serde_json::json!({
        "subject": "Charged twice",
        "order": {"id": 4411, "paid": true},
        "notes": ["refund one", "keep the other"],
    }))
    .unwrap()
    .ask("billing", Noul::new("Is this about billing?"));

    engine.system_one_blocking(&request).unwrap();

    assert_eq!(
        log.lock().unwrap().texts[0],
        "subject: Charged twice\norder:\n  id: 4411\n  paid: True\nnotes:\n  - refund one\n  - keep the other"
    );
}

#[test]
fn the_readout_reads_the_decide_and_end_of_option_tokens() {
    let (engine, log, _) = engine(readme_targets());

    engine.system_one_blocking(&readme_request()).unwrap();

    let log = log.lock().unwrap();
    let pass = &log.passes[0];
    let read: Vec<u32> = pass.readout.iter().map(|index| pass.ids[*index]).collect();
    assert_eq!(
        read,
        [
            DECIDE_ID,
            OPTION_END_ID,
            OPTION_END_ID,
            OPTION_END_ID, // department
            DECIDE_ID,
            OPTION_END_ID,
            OPTION_END_ID, // escalate
            DECIDE_ID,
            OPTION_END_ID,
            OPTION_END_ID,
            OPTION_END_ID, // frustration
        ]
    );
}

#[test]
fn a_question_can_read_the_state_but_not_another_question() {
    let (engine, log, _) = engine(readme_targets());

    engine.system_one_blocking(&readme_request()).unwrap();

    let log = log.lock().unwrap();
    let recorded = &log.passes[0];
    let pass = recorded.as_pass();
    let (first, second) = (recorded.branch(1), recorded.branch(2));

    for token in &second {
        for earlier in &first {
            assert!(
                !pass.attends(*token, *earlier),
                "token {token} of question 2 can read token {earlier} of question 1"
            );
        }
        assert!(pass.attends(*token, 0), "question 2 cannot read the state");
        assert!(
            !pass.attends(*token, token + 1) || *token + 1 >= recorded.ids.len(),
            "question 2 can read a later token"
        );
    }
    assert!(!pass.is_causal(), "three questions need the branch mask");
}

#[test]
fn every_question_is_positioned_as_if_it_were_the_only_one() {
    let (engine, log, _) = engine(readme_targets());

    engine.system_one_blocking(&readme_request()).unwrap();

    let log = log.lock().unwrap();
    let recorded = &log.passes[0];
    let state_len = recorded.branch(0).len();
    for question in 1..=3 {
        let branch = recorded.branch(question);
        let positions: Vec<u32> = branch
            .iter()
            .map(|index| recorded.positions[*index])
            .collect();
        let expected: Vec<u32> =
            (state_len as u32..state_len as u32 + branch.len() as u32).collect();
        assert_eq!(
            positions, expected,
            "question {question} sits at the wrong positions"
        );
    }
}

#[test]
fn asking_separately_gives_the_same_answers_one_pass_at_a_time() {
    let (packed, _, _) = engine(readme_targets());
    let (separate, log, passes) = engine(readme_targets());

    let together = packed.system_one_blocking(&readme_request()).unwrap();
    let apart = separate
        .system_one_separate_blocking(&readme_request())
        .unwrap();

    assert_eq!(passes.load(Ordering::SeqCst), 3, "one pass per question");
    for id in ["department", "escalate", "frustration"] {
        assert_eq!(
            format!("{:?}", apart.answer(id).unwrap()),
            format!("{:?}", together.answer(id).unwrap()),
            "{id} differs between the packed and the separate call"
        );
    }
    // Each pass carries the state and exactly one question.
    let log = log.lock().unwrap();
    for pass in &log.passes {
        assert!(
            pass.as_pass().is_causal(),
            "a separate pass needs no branch mask"
        );
    }
    // The state is encoded once per question, and billed that way.
    assert!(apart.usage.input_tokens > together.usage.input_tokens);
}

#[test]
fn a_backend_that_can_only_run_causal_rows_gets_the_same_answers() {
    let (stub, _, passes) = Stub::new(readme_targets());
    let rows = LocalEngine::new(Stub { rows: true, ..stub }, identity_head());
    let (packed, _, _) = engine(readme_targets());

    let through_rows = rows.system_one_blocking(&readme_request()).unwrap();
    let packed = packed.system_one_blocking(&readme_request()).unwrap();

    assert_eq!(
        passes.load(Ordering::SeqCst),
        1,
        "still one call into the backend"
    );
    for id in ["department", "escalate", "frustration"] {
        assert_eq!(
            format!("{:?}", through_rows.answer(id).unwrap()),
            format!("{:?}", packed.answer(id).unwrap()),
            "{id} differs between the packed and the row form"
        );
    }
}

#[test]
fn the_answers_are_billed_the_way_the_server_serialises_them() {
    let (engine, log, _) = engine(readme_targets());

    let response = engine.system_one_blocking(&readme_request()).unwrap();

    // `output_tokens` counts the serialised answers, not generated tokens -
    // there are none. This stub bills one token per byte.
    let serialised = log.lock().unwrap().texts.last().unwrap().clone();
    assert_eq!(
        serialised,
        "{\"department\": {\"type\": \"choice\", \"choice\": \"returns\", \"confidence\": 0.205, \
         \"probabilities\": {\"returns\": 0.47, \"shipping\": 0.28, \"billing\": 0.25}}, \
         \"escalate\": {\"type\": \"noul\", \"noul\": 0.93}, \
         \"frustration\": {\"type\": \"score\", \"score\": 1.44, \
         \"legend\": {\"0\": \"Calm\", \"1\": \"Frustrated\", \"2\": \"Very angry\"}, \
         \"probabilities\": {\"0\": 0.0, \"1\": 0.56, \"2\": 0.44}, \"confidence\": 0.78}}"
    );
    assert_eq!(response.usage.output_tokens, serialised.len() as u64);
    assert!(response.usage.input_tokens > 0);
    assert!(response.latency_ms.is_some());
    // A header the HTTP server sets; there is none here.
    assert_eq!(response.request_id, None);
}

#[test]
fn the_model_is_the_one_the_request_pins_or_the_engine_default() {
    let (engine, _, _) = engine(vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    let ask = || SystemOneRequest::new("x").ask("q", Noul::new("?"));

    let default = engine.system_one_blocking(&ask()).unwrap();
    let pinned = engine.system_one_blocking(&ask().model("kev-4b")).unwrap();

    assert_eq!(default.model, kev_client::DEFAULT_MODEL);
    assert_eq!(pinned.model, "kev-4b");
}

#[test]
fn a_question_too_long_for_the_context_is_a_validation_error() {
    let (stub, _, _) = Stub::new(vec![vec![0.5, 0.5]]);
    let engine = LocalEngine::new(stub, identity_head()).with_limits(Limits {
        max_state: 16,
        max_branch: 32,
    });
    let long = Choice::new("Pick one")
        .options((0..20).map(|i| (format!("option-{i}"), "a description that costs tokens")));

    let error = engine
        .system_one_blocking(&SystemOneRequest::new("a ticket").ask("q", long))
        .unwrap_err();

    assert!(error.is_validation(), "got {error:?}");
    assert!(error.to_string().contains("context"), "{error}");
}

#[test]
fn a_state_longer_than_the_context_is_truncated_rather_than_refused() {
    let (stub, log, _) = Stub::new(vec![vec![0.5, 0.5]]);
    let engine = LocalEngine::new(stub, identity_head()).with_limits(Limits {
        max_state: 8,
        max_branch: 64,
    });

    let response = engine.system_one_blocking(
        &SystemOneRequest::new("a state far longer than eight tokens").ask("q", Noul::new("?")),
    );

    assert!(response.is_ok(), "{:?}", response.unwrap_err());
    let log = log.lock().unwrap();
    let state = log.passes[0].branch(0);
    assert_eq!(state.len(), 8, "the state kept its delimiter and 7 tokens");
}

#[test]
fn clones_share_one_backend_rather_than_loading_the_model_twice() {
    let (engine, _, passes) = engine(vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    let clone = engine.clone();
    let ask = || SystemOneRequest::new("x").ask("q", Noul::new("?"));

    engine.system_one_blocking(&ask()).unwrap();
    clone.system_one_blocking(&ask()).unwrap();

    assert_eq!(passes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn the_engine_answers_through_the_same_seam_as_the_http_client() {
    // The point of the seam: this function never names LocalEngine.
    async fn ask<B: SystemOne>(backend: &B) -> Result<f64> {
        let response = backend
            .system_one(&SystemOneRequest::new("I was charged twice.").ask(
                "billing",
                Noul::new("Is this ticket about billing?").yes("it is about money"),
            ))
            .await?;
        Ok(response.answer("billing").unwrap().as_noul().unwrap())
    }

    let (engine, _, _) = engine(vec![vec![0.2, 0.8]]);

    assert_eq!(ask(&engine).await.unwrap(), 0.8);
}
