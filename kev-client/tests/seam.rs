//! The backend seam. `Client` is one way to answer a System One request; a
//! local inference engine will be another. These tests pin what the seam has
//! to allow, and need no server.

use std::future::Future;

use kev_client::{
    Answer, IndexMap, Noul, Result, SystemOne, SystemOneRequest, SystemOneResponse, Usage,
};

/// A backend that answers from a canned response, the way a local engine will:
/// no HTTP involved. Compiles and runs with `--no-default-features`.
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
        request_id: None,
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
async fn a_backend_that_is_not_http_can_answer_through_the_seam() {
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

#[cfg(feature = "http")]
mod over_http {
    use super::a_request;
    use kev_client::{Client, SystemOne};

    fn assert_send<F: Send>(_future: F) {}

    #[test]
    fn the_http_client_implements_the_seam() {
        fn triage<B: SystemOne>(_backend: &B) {}

        triage(&Client::local().unwrap());
    }

    #[test]
    fn the_returned_futures_can_be_sent_across_threads() {
        // The `+ Send` in the trait is what lets a caller tokio::spawn a
        // request. Building a future sends nothing, so this needs no server.
        let client = Client::local().unwrap();
        let request = a_request();

        assert_send(SystemOne::system_one(&client, &request));
        assert_send(SystemOne::system_one_separate(&client, &request));
    }

    #[test]
    fn the_inherent_methods_still_win_for_existing_callers() {
        // Client keeps its own system_one, so importing the trait cannot
        // change which method existing code calls.
        let client = Client::local().unwrap();

        assert_send(client.system_one(&a_request()));
    }
}
