//! The backend seam. `LocalEngine` is one way to answer a System One request; a
//! recording, a queue or another engine can be others. These tests pin what the
//! seam has to allow, and the first of them runs with no features at all.

use std::future::Future;

// The stub backend, shared with tests/local_engine.rs. At the root of this test
// crate rather than inside the module that uses it: a #[path] resolves relative
// to the module it sits in.
#[cfg(feature = "local")]
#[path = "fixtures/mod.rs"]
mod fixtures;

use kev_client::{
    Answer, IndexMap, Noul, Result, SystemOne, SystemOneRequest, SystemOneResponse, Usage,
};

/// A backend that answers from a canned response. Compiles and runs with
/// `--no-default-features`, which is the point: the seam costs a caller nothing.
struct Canned(SystemOneResponse);

// Kept as `-> impl Future + Send` rather than `async fn`: it mirrors the trait
// signature exactly, so the Send bound stays visible at every implementation
// site instead of being something the compiler infers out of sight.
#[allow(clippy::manual_async_fn)]
impl SystemOne for Canned {
    fn system_one(
        &self,
        _request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send {
        async move { Ok(self.0.clone()) }
    }

    fn system_one_separate(
        &self,
        request: &SystemOneRequest,
    ) -> impl Future<Output = Result<SystemOneResponse>> + Send {
        self.system_one(request)
    }
}

fn canned_response() -> SystemOneResponse {
    let mut answers = IndexMap::new();
    answers.insert("billing".to_string(), Answer::Noul { noul: 0.9 });

    SystemOneResponse {
        model: "stub".to_string(),
        answers,
        usage: Usage::default(),
        latency_ms: None,
    }
}

fn a_request() -> SystemOneRequest {
    SystemOneRequest::new("I was charged twice.").ask("billing", Noul::new("Billing problem?"))
}

/// What the seam is for: this function names no backend type at all.
async fn ask<B: SystemOne>(backend: &B, request: &SystemOneRequest) -> Result<SystemOneResponse> {
    backend.system_one(request).await
}

#[tokio::test]
async fn a_backend_with_no_model_in_it_can_answer_through_the_seam() {
    let backend = Canned(canned_response());

    let response = ask(&backend, &a_request()).await.unwrap();

    assert_eq!(response.answer("billing").unwrap().as_noul(), Some(0.9));
}

#[tokio::test]
async fn the_separate_call_goes_through_the_same_seam() {
    let backend = Canned(canned_response());

    let response = backend.system_one_separate(&a_request()).await.unwrap();

    assert_eq!(response.model, "stub");
}

#[cfg(feature = "local")]
mod over_the_local_engine {
    use super::{a_request, fixtures};
    use kev_client::SystemOne;

    fn assert_send<F: Send>(_future: F) {}

    #[test]
    fn the_local_engine_implements_the_seam() {
        fn triage<B: SystemOne>(_backend: &B) {}

        let (engine, _log, _calls) = fixtures::engine(vec![vec![1.0, 0.0]]);
        triage(&engine);
    }

    #[test]
    fn the_returned_futures_can_be_sent_across_threads() {
        // The `+ Send` in the trait is what lets a caller tokio::spawn a request.
        // Building a future runs nothing, so this needs no weights.
        let (engine, _log, _calls) = fixtures::engine(vec![vec![1.0, 0.0]]);
        let request = a_request();

        assert_send(SystemOne::system_one(&engine, &request));
        assert_send(SystemOne::system_one_separate(&engine, &request));
    }

    #[test]
    fn the_inherent_methods_still_win_for_existing_callers() {
        // The engine keeps its own system_one, so importing the trait cannot
        // change which method existing code calls.
        let (engine, _log, _calls) = fixtures::engine(vec![vec![1.0, 0.0]]);

        assert_send(engine.system_one(&a_request()));
    }
}
