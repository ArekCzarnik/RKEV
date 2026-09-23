//! The Python kev's own unit tests, ported.
//!
//! `tests/test_unit.py` upstream pins the API mapping, the confidence formulas
//! and the mask rule with exact values, and those expectations port over
//! unchanged. They are a second opinion on the readout that does not come from
//! this repository — which is worth having while the parity run against a real
//! checkpoint is still ahead of us.
//!
//! Each test names the one it comes from.

#![cfg(feature = "local")]

mod fixtures;

use fixtures::engine;
use kev_client::{render, Choice, Linear, Noul, Pass, PointerHead, Score, SystemOneRequest};
use serde_json::json;

/// `test_render_flattens_structured_content`
#[test]
fn render_flattens_structured_content() {
    assert_eq!(render(&json!("plain")), "plain");
    assert_eq!(render(&json!(null)), "");
    assert_eq!(
        render(&json!({"what": "A", "not_for": "B"})),
        "what: A\nnot_for: B"
    );
    assert_eq!(render(&json!(["x", "y"])), "- x\n- y");
    assert_eq!(
        render(&json!({"ticket": {"channel": "email", "body": "hi"}})),
        "ticket:\n  channel: email\n  body: hi"
    );
    assert_eq!(
        render(&json!({"examples": ["a", "b"]})),
        "examples:\n  - a\n  - b"
    );
}

/// `test_to_record_maps_all_three_types`: the same state text, the same options
/// in the same order, and the same keys to read the answers back under.
#[test]
fn a_request_maps_to_the_record_the_python_builds() {
    let request = SystemOneRequest::from_state(&json!({"document": "I was charged twice."}))
        .unwrap()
        .ask(
            "billing",
            Noul::new("About billing?").yes("Charges").no("Not charges"),
        )
        .ask(
            "tone",
            Choice::new("Tone?")
                .option_bare("calm")
                .option("angry", "Hostile"),
        )
        .ask(
            "urgency",
            Score::new("Urgency?").level("can wait").level("today"),
        );
    let (engine, log, _) = engine(vec![vec![0.4, 0.6], vec![0.5, 0.5], vec![0.5, 0.5]]);

    let response = engine.system_one_blocking(&request).unwrap();

    let texts = log.lock().unwrap().texts.clone();
    assert_eq!(texts[0], "document: I was charged twice.");
    // Per question: the instructions, then the options.
    assert_eq!(
        texts[1..4],
        ["About billing?", "no: Not charges", "yes: Charges"]
    );
    assert_eq!(texts[4..7], ["Tone?", "calm", "angry: Hostile"]);
    assert_eq!(texts[7..10], ["Urgency?", "can wait", "today"]);

    let ids: Vec<&str> = response.answers.keys().map(String::as_str).collect();
    assert_eq!(ids, ["billing", "tone", "urgency"]);
    let tone = response.answer("tone").unwrap();
    let keys: Vec<&str> = tone
        .probabilities()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["calm", "angry"]);
    let urgency = response.answer("urgency").unwrap();
    assert_eq!(urgency.legend().unwrap()["0"], "can wait");
    assert_eq!(urgency.legend().unwrap()["1"], "today");
}

/// `test_to_answers_shapes_and_formulas`
#[test]
fn answers_have_the_shapes_and_formulas_the_python_gives() {
    let request = SystemOneRequest::new("s")
        .ask("n", Noul::new("i"))
        .ask(
            "c",
            Choice::new("i")
                .option_bare("a")
                .option_bare("b")
                .option_bare("c"),
        )
        .ask("s", Score::new("i").level("lo").level("mid").level("hi"));
    let (engine, _, _) = engine(vec![
        vec![0.3, 0.7],
        vec![0.8, 0.15, 0.05],
        vec![0.1, 0.3, 0.6],
    ]);

    let response = engine.system_one_blocking(&request).unwrap();

    assert_eq!(response.answer("n").unwrap().as_noul(), Some(0.7));
    let choice = response.answer("c").unwrap();
    assert_eq!(choice.as_choice(), Some("a"));
    assert_eq!(choice.probabilities().unwrap()["a"], 0.8);
    assert_eq!(choice.probabilities().unwrap()["b"], 0.15);
    assert_eq!(choice.probabilities().unwrap()["c"], 0.05);
    // round((0.8 - 1/3) / (1 - 1/3), 4)
    assert_eq!(choice.confidence(), Some(0.7));
    let score = response.answer("s").unwrap();
    assert_eq!(score.as_score(), Some(1.5));
    assert_eq!(score.probabilities().unwrap()["2"], 0.6);
    assert_eq!(score.legend().unwrap()["1"], "mid");
}

/// `test_to_answers_choice_probabilities_sum_within_typesafe_tolerance`: four
/// decimals keep a rounded distribution within `|sum - 1| < 0.02`, even at the
/// 255-option maximum.
#[test]
fn a_rounded_distribution_still_sums_to_one_closely_enough() {
    for probabilities in [
        std::iter::once(0.79)
            .chain(std::iter::repeat_n(0.21 / 39.0, 39))
            .collect::<Vec<f64>>(),
        vec![1.0 / 255.0; 255],
    ] {
        let mut question = Choice::new("i");
        for index in 0..probabilities.len() {
            question = question.option_bare(format!("o{index}"));
        }
        let (engine, _, _) = engine(vec![probabilities.clone()]);

        let response = engine
            .system_one_blocking(&SystemOneRequest::new("s").ask("target", question))
            .unwrap();

        let served = response.answer("target").unwrap().probabilities().unwrap();
        assert_eq!(served.len(), probabilities.len());
        let sum: f64 = served.values().sum();
        assert!(
            (sum - 1.0).abs() < 0.02,
            "{} options summed to {sum}",
            probabilities.len()
        );
    }
}

/// `test_confidence_edge_cases`. The probabilities a stub can produce are
/// softmaxes, so a flat zero is written as very nearly zero — which is what a
/// model would give anyway.
#[test]
fn the_confidence_edge_cases_are_the_python_ones() {
    let ask = |targets: Vec<f64>, options: usize, score: bool| {
        let (engine, _, _) = engine(vec![targets]);
        let question = if score {
            let mut question = Score::new("i");
            for index in 0..options {
                question = question.level(format!("level {index}"));
            }
            engine
                .system_one_blocking(&SystemOneRequest::new("s").ask("q", question))
                .unwrap()
        } else {
            let mut question = Choice::new("i");
            for index in 0..options {
                question = question.option_bare(format!("o{index}"));
            }
            engine
                .system_one_blocking(&SystemOneRequest::new("s").ask("q", question))
                .unwrap()
        };
        question.answer("q").unwrap().confidence().unwrap()
    };

    // A single option is certain by construction, for a choice and for a score.
    assert_eq!(ask(vec![1.0], 1, false), 1.0);
    assert_eq!(ask(vec![1.0], 1, true), 1.0);
    // Two options, evenly split: no better than guessing.
    assert_eq!(ask(vec![0.5, 0.5], 2, false), 0.0);
    // All on one option, and all on one level.
    assert!((ask(vec![0.9999, 0.00005, 0.00005], 3, false) - 1.0).abs() < 1e-3);
    assert_eq!(ask(vec![0.00002, 0.99996, 0.00002], 3, true), 1.0);
    // Split across the ends of a scale: somewhere in range, and not certain.
    let split = ask(vec![0.5, 0.00002, 0.49998], 3, true);
    assert!((0.0..=1.0).contains(&split), "{split}");
}

/// `test_branch_mask_rule`, with the same segments and the same indices.
#[test]
fn a_question_sees_the_state_and_itself_by_the_python_rule() {
    let ids = [0u32; 6];
    let positions = [0u32, 1, 2, 3, 2, 3];
    let segments = [0u32, 0, 1, 1, 2, 2];
    let pass = Pass {
        ids: &ids,
        positions: &positions,
        segments: &segments,
        readout: &[],
    };

    // Question 1 sees the state and itself.
    assert!(pass.attends(3, 0) && pass.attends(3, 1) && pass.attends(3, 2));
    // Not the future.
    assert!(!pass.attends(3, 4) && !pass.attends(3, 5));
    // Question 2 never sees question 1.
    assert!(pass.attends(5, 0) && pass.attends(5, 4));
    assert!(!pass.attends(5, 2) && !pass.attends(5, 3));
    // The state is causal too.
    assert!(!pass.attends(0, 1));
}

/// `test_head_temperature_scales_logits_at_eval_only`: the temperature divides
/// the logits and cannot change which option wins.
#[test]
fn the_temperature_divides_the_logits_and_leaves_the_winner() {
    let hidden = 4;
    let weight: Vec<f32> = (0..3 * hidden).map(|i| (i as f32 - 5.0) / 3.0).collect();
    let bias = vec![0.1, -0.2, 0.3];
    let projection = || Linear::new(weight.clone(), bias.clone(), hidden).unwrap();
    let raw = PointerHead::new(projection(), projection()).unwrap();
    let calibrated = PointerHead::new(projection(), projection())
        .unwrap()
        .with_temperature(2.0)
        .unwrap();

    let decide = vec![0.3, -0.7, 1.1, 0.5];
    let options = vec![vec![1.0, 0.2, -0.4, 0.8], vec![-0.6, 0.9, 0.1, -0.3]];
    let raw = raw.logits(&decide, &options).unwrap();
    let calibrated = calibrated.logits(&decide, &options).unwrap();

    for (raw, calibrated) in raw.iter().zip(&calibrated) {
        assert!(
            (raw / 2.0 - calibrated).abs() < 1e-6,
            "{raw} vs {calibrated}"
        );
    }
    let winner = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
    };
    assert_eq!(winner(&raw), winner(&calibrated));
}
