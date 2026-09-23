//! The local engine skeleton. No model is loaded — a stub backend stands in
//! for mistral.rs — so these tests pin the plumbing, not the answers: the
//! engine sits on the same seam as the HTTP client, shares one backend across
//! clones, and says clearly that it cannot answer yet instead of panicking.

#![cfg(feature = "local")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kev_client::{Error, Forward, LocalEngine, Noul, Result, SystemOne, SystemOneRequest};

/// Stands in for mistral.rs: records what it was asked, answers with zeros.
struct StubBackend {
    calls: Arc<AtomicUsize>,
}

impl Forward for StubBackend {
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(text.bytes().map(u32::from).collect())
    }

    fn logits(&mut self, _tokens: &[u32], positions: &[usize]) -> Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(positions.iter().map(|_| vec![0.0; 8]).collect())
    }
}

fn an_engine() -> (LocalEngine, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = LocalEngine::new(StubBackend {
        calls: Arc::clone(&calls),
    });
    (engine, calls)
}

fn a_request() -> SystemOneRequest {
    SystemOneRequest::new("I was charged twice.").ask("billing", Noul::new("Billing problem?"))
}

#[test]
fn the_engine_reports_that_it_cannot_answer_yet() {
    // Until the readout lands, callers get an error they can act on rather
    // than a panic or a plausible-looking wrong distribution.
    let (engine, _) = an_engine();

    let error = engine.system_one_blocking(&a_request()).unwrap_err();

    assert!(matches!(error, Error::Engine(_)), "got {error:?}");
    assert!(error.to_string().contains("forward pass"), "{error}");
}

#[test]
fn the_engine_reaches_its_backend() {
    let (engine, calls) = an_engine();

    let _ = engine.system_one_blocking(&a_request());

    assert!(calls.load(Ordering::SeqCst) > 0, "backend was never called");
}

#[test]
fn clones_share_one_backend_rather_than_loading_the_model_twice() {
    let (engine, calls) = an_engine();
    let clone = engine.clone();

    let _ = engine.system_one_blocking(&a_request());
    let _ = clone.system_one_blocking(&a_request());

    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn the_engine_answers_through_the_same_seam_as_the_http_client() {
    // The point of the seam: this function never names LocalEngine.
    async fn ask<B: SystemOne>(backend: &B) -> Result<()> {
        backend.system_one(&a_request()).await.map(|_| ())
    }

    let (engine, _) = an_engine();

    let error = ask(&engine).await.unwrap_err();

    assert!(matches!(error, Error::Engine(_)), "got {error:?}");
}
