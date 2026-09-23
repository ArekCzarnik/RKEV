use std::fmt;

/// Everything that can go wrong answering a System One request.
#[derive(Debug)]
pub enum Error {
    /// The request itself does not make sense — what the Python server answers
    /// `422` for.
    Invalid(String),
    /// A question did not fit the context: the state plus that one question
    /// exceeded the row limit. The Python server answers `422` for this, so
    /// [`Error::is_validation`] is true here too.
    #[cfg(feature = "local")]
    ContextOverflow(String),
    /// The local engine failed: loading the model, tokenising, or the
    /// forward pass itself.
    #[cfg(feature = "local")]
    Engine(String),
    /// The backbone failed: loading weights, or the forward pass itself.
    #[cfg(feature = "candle")]
    Model(candle_core::Error),
}

impl Error {
    /// `true` for what the Python server would answer `422` to: a malformed
    /// request, or one that does not fit the context.
    pub fn is_validation(&self) -> bool {
        #[cfg(feature = "local")]
        if matches!(self, Error::ContextOverflow(_)) {
            return true;
        }
        matches!(self, Error::Invalid(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message) => write!(f, "invalid request: {message}"),
            #[cfg(feature = "local")]
            Error::ContextOverflow(message) => {
                write!(f, "the request does not fit the model context: {message}")
            }
            #[cfg(feature = "local")]
            Error::Engine(message) => write!(f, "the local kev engine failed: {message}"),
            #[cfg(feature = "candle")]
            Error::Model(e) => write!(f, "the model failed: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "candle")]
            Error::Model(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "candle")]
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Model(e)
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
