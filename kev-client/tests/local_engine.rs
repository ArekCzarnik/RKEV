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

mod fixtures;

use std::sync::atomic::Ordering;

use fixtures::{engine, identity_head, Stub, DECIDE_ID, OPTION_END_ID};
use kev_client::{
    Choice, Limits, Linear, LocalEngine, Noul, PointerHead, Result, Score, SystemOne,
    SystemOneRequest,
};

/// Which projection reads `<decide>` and which reads each `</opt>` is decided by
/// `head.pt`'s names, and nothing about the shapes says so — swapped, the head
/// scores just as plausibly. So the swap has to be a real difference (something an
/// obvious case can then separate on a trained checkpoint) and it has to be an
/// involution, or the check built on it means nothing.
#[test]
fn swapping_the_pointer_heads_projections_changes_the_scores_and_reverses() {
    // q is the identity, k adds the second unit to the first: asymmetric, so the
    // swap cannot be hidden by symmetry.
    let q = Linear::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2).unwrap();
    let k = Linear::new(vec![1.0, 1.0, 0.0, 1.0], vec![0.0, 0.0], 2).unwrap();
    let head = PointerHead::new(q, k).unwrap();
    let decide = [1.0, 2.0];
    let options = [vec![1.0, 0.0], vec![0.0, 1.0]];

    let straight = head.clone().logits(&decide, &options).unwrap();
    let swapped = head.clone().swapped().logits(&decide, &options).unwrap();
    let twice = head
        .clone()
        .swapped()
        .swapped()
        .logits(&decide, &options)
        .unwrap();

    let scale = 1.0 / 2.0f32.sqrt();
    // q(decide) = [1, 2]; k(o1) = [1, 0], k(o2) = [1, 1].
    assert_eq!(straight, vec![1.0 * scale, 3.0 * scale]);
    // k(decide) = [3, 2]; q(o1) = [1, 0], q(o2) = [0, 1]. A different winner, too.
    assert_eq!(swapped, vec![3.0 * scale, 2.0 * scale]);
    assert_eq!(twice, straight, "swapping twice has to be the head itself");
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
async fn the_engine_answers_through_the_same_seam() {
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

#[test]
fn a_batch_of_requests_is_answered_like_a_sequence_of_them() {
    // Over a backend that does not override `hidden_batch`, the batch API is the
    // loop it would have been - which is what keeps it usable for any backend,
    // not just the one that can share a prefill.
    let (engine, _, passes) = engine(vec![vec![0.3, 0.7], vec![0.6, 0.4], vec![0.55, 0.45]]);
    let requests: Vec<SystemOneRequest> = ["a ticket", "another ticket", "a third"]
        .iter()
        .map(|state| SystemOneRequest::new(*state).ask("q", Noul::new("about money?")))
        .collect();

    let answers = engine.system_one_batch_blocking(&requests).unwrap();

    assert_eq!(answers.len(), 3);
    assert_eq!(answers[0].answer("q").unwrap().as_noul(), Some(0.7));
    assert_eq!(answers[1].answer("q").unwrap().as_noul(), Some(0.4));
    assert_eq!(answers[2].answer("q").unwrap().as_noul(), Some(0.45));
    assert_eq!(passes.load(Ordering::SeqCst), 3, "one pass per request");
}

#[test]
fn a_request_that_does_not_fit_fails_the_whole_batch() {
    let (stub, _, _) = Stub::new(vec![vec![0.5, 0.5]]);
    let engine = LocalEngine::new(stub, identity_head()).with_limits(Limits {
        max_state: 16,
        max_branch: 32,
    });
    let long = Choice::new("Pick one")
        .options((0..20).map(|i| (format!("option-{i}"), "a description that costs tokens")));

    let error = engine
        .system_one_batch_blocking(&[
            SystemOneRequest::new("fine").ask("q", Noul::new("?")),
            SystemOneRequest::new("a ticket").ask("q", long),
        ])
        .unwrap_err();

    assert!(error.is_validation(), "got {error:?}");
}

#[test]
fn permuting_runs_the_orders_and_reports_what_moved() {
    // The stub answers by option *position*, so it is a model that cares about
    // order — which is what this endpoint exists to detect.
    let (engine, log, _) = engine(vec![vec![0.7, 0.2, 0.1]; 6]);
    let request = SystemOneRequest::new("a ticket").ask(
        "team",
        Choice::new("which team ?")
            .option_bare("returns")
            .option_bare("shipping")
            .option_bare("billing"),
    );

    let permuted = engine.permute_blocking(&request, "team", 6, 0).unwrap();

    let runs = permuted["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 6);
    // The first run keeps the order as given.
    assert_eq!(
        runs[0]["order"].as_array().unwrap(),
        &["returns", "shipping", "billing"]
            .map(serde_json::Value::from)
            .to_vec()
    );
    // Every run reports a full distribution and its winner.
    for run in runs {
        assert_eq!(run["probabilities"].as_object().unwrap().len(), 3);
        assert!(run["choice"].is_string());
        assert!(run["latency_ms"].is_f64());
    }
    // A model that answers by position cannot be stable under shuffling.
    assert_eq!(permuted["argmax_stable"], serde_json::json!(false));
    let spread = permuted["spread"].as_object().unwrap();
    assert_eq!(spread.len(), 3);
    assert!(
        spread.values().any(|value| value.as_f64().unwrap() > 0.5),
        "{spread:?}"
    );
    // Only the named question was asked, six times over the same state.
    let texts = &log.lock().unwrap().texts;
    assert_eq!(texts.iter().filter(|text| *text == "a ticket").count(), 6);
    assert!(!texts.iter().any(|text| text == "is this late ?"));
}

#[test]
fn one_round_of_permuting_moves_nothing() {
    let (engine, _, _) = engine(vec![vec![0.6, 0.4]]);
    let request = SystemOneRequest::new("a ticket").ask(
        "team",
        Choice::new("which team ?")
            .option_bare("returns")
            .option_bare("billing"),
    );

    // Zero rounds is one round: the clamp the client applies to `n_perm`.
    let permuted = engine.permute_blocking(&request, "team", 0, 7).unwrap();

    assert_eq!(permuted["runs"].as_array().unwrap().len(), 1);
    assert_eq!(permuted["argmax_stable"], serde_json::json!(true));
    for spread in permuted["spread"].as_object().unwrap().values() {
        assert_eq!(spread.as_f64().unwrap(), 0.0);
    }
}

#[test]
fn permuting_is_the_same_twice_from_the_same_seed() {
    let orders = |seed: u64| {
        let (engine, _, _) = engine(vec![vec![0.5, 0.3, 0.2]; 5]);
        let request = SystemOneRequest::new("a ticket").ask(
            "team",
            Choice::new("which team ?")
                .option_bare("returns")
                .option_bare("shipping")
                .option_bare("billing"),
        );
        let permuted = engine.permute_blocking(&request, "team", 5, seed).unwrap();
        permuted["runs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|run| run["order"].clone())
            .collect::<Vec<_>>()
    };

    assert_eq!(orders(3), orders(3));
    assert_ne!(orders(3), orders(4), "the seed has to change something");
}

#[test]
fn permuting_anything_but_a_choice_question_is_a_validation_error() {
    let (engine, _, _) = engine(vec![vec![0.5, 0.5]]);
    let request = SystemOneRequest::new("a ticket").ask("late", Noul::new("is this late ?"));

    let error = engine.permute_blocking(&request, "late", 4, 0).unwrap_err();
    assert!(error.is_validation(), "got {error:?}");
    assert!(error.to_string().contains("choice question"), "{error}");

    let missing = engine
        .permute_blocking(&request, "absent", 4, 0)
        .unwrap_err();
    assert!(missing.is_validation(), "got {missing:?}");
}

#[tokio::test]
async fn permuting_works_off_the_runtime_thread_too() {
    let (engine, _, _) = engine(vec![vec![0.6, 0.4]; 3]);
    let request = SystemOneRequest::new("a ticket").ask(
        "team",
        Choice::new("which team ?")
            .option_bare("returns")
            .option_bare("billing"),
    );

    let permuted = engine.permute(&request, "team", 3, 1).await.unwrap();

    assert_eq!(permuted["runs"].as_array().unwrap().len(), 3);
}

#[test]
fn isolating_the_options_puts_every_span_at_the_same_positions() {
    use kev_client::OptionSlot;

    // Every option span restarts where the instructions end, and `<decide>` sits
    // past the longest of them — so no option can be told apart by where it sits.
    let (stub, log, _) = Stub::new(vec![vec![0.5, 0.3, 0.2]]);
    let engine = LocalEngine::new(stub, identity_head()).with_option_isolation(true);
    let request = SystemOneRequest::new("a ticket").ask(
        "team",
        Choice::new("which team ?")
            .option_bare("returns")
            .option("shipping", "a much longer description")
            .option_bare("x"),
    );

    engine.system_one_blocking(&request).unwrap();

    let log = log.lock().unwrap();
    let pass = &log.passes[0];
    assert_eq!(pass.options.len(), pass.ids.len(), "every token has a slot");

    let span = |option: u8| -> Vec<u32> {
        pass.options
            .iter()
            .zip(&pass.positions)
            .filter(|(slot, _)| **slot == OptionSlot::Option(option))
            .map(|(_, position)| *position)
            .collect()
    };
    let starts: Vec<u32> = (0..3).map(|option| span(option)[0]).collect();
    assert_eq!(starts[0], starts[1], "spans start at the same position");
    assert_eq!(starts[1], starts[2]);
    // Each span is consecutive from there.
    for option in 0..3 {
        let positions = span(option);
        let expected: Vec<u32> = (starts[0]..starts[0] + positions.len() as u32).collect();
        assert_eq!(positions, expected, "option {option}");
    }
    // `<decide>` sits one past the longest span, whichever option that was.
    let longest = (0..3).map(|option| span(option).len()).max().unwrap() as u32;
    let decide = pass
        .options
        .iter()
        .zip(&pass.positions)
        .find(|(slot, _)| **slot == OptionSlot::Decide)
        .map(|(_, position)| *position)
        .unwrap();
    assert_eq!(decide, starts[0] + longest);
}

#[test]
fn without_isolation_no_token_carries_a_slot() {
    let (engine, log, _) = engine(vec![vec![0.5, 0.5]]);

    engine
        .system_one_blocking(&SystemOneRequest::new("a ticket").ask("q", Noul::new("late?")))
        .unwrap();

    // The usual layout costs nothing to carry: the slots stay empty, and
    // `Pass::attends` never looks at them.
    assert!(log.lock().unwrap().passes[0].options.is_empty());
    assert!(!log.lock().unwrap().passes[0].as_pass().is_isolated());
}
