use std::fmt;

/// Everything that can go wrong when talking to a Kev server.
#[derive(Debug)]
pub enum Error {
    /// The base URL was not an absolute `http://` or `https://` URL.
    InvalidBaseUrl(String),
    /// The request never completed (connection refused, timeout, TLS, ...).
    #[cfg(feature = "http")]
    Transport(reqwest::Error),
    /// The server answered with a non-2xx status. Kev returns `422` for
    /// invalid requests, `401` when `KEV_API_KEY` is set and the bearer
    /// token is missing or wrong.
    Api {
        status: u16,
        /// Value of the `x-typesafe-request-id` response header, if present.
        request_id: Option<String>,
        body: String,
    },
    /// The response was 2xx but did not match the expected shape.
    Decode {
        source: serde_json::Error,
        body: String,
    },
}

impl Error {
    /// `true` for the `422` that Kev returns for a malformed request body.
    pub fn is_validation(&self) -> bool {
        matches!(self, Error::Api { status: 422, .. })
    }

    /// `true` when the server demands a bearer token (`KEV_API_KEY` is set).
    pub fn is_unauthorized(&self) -> bool {
        matches!(self, Error::Api { status: 401 | 403, .. })
    }

    /// The server-side request id, for correlating with the server log.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            Error::Api { request_id, .. } => request_id.as_deref(),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidBaseUrl(url) => {
                write!(f, "base url must start with http:// or https://, got {url:?}")
            }
            #[cfg(feature = "http")]
            Error::Transport(e) => write!(f, "request to the kev server failed: {e}"),
            Error::Api {
                status,
                request_id,
                body,
            } => {
                write!(f, "kev server returned HTTP {status}")?;
                if let Some(id) = request_id {
                    write!(f, " (request id {id})")?;
                }
                write!(f, ": {body}")
            }
            Error::Decode { source, body } => {
                write!(f, "could not decode the kev response ({source}): {body}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "http")]
            Error::Transport(e) => Some(e),
            Error::Decode { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(feature = "http")]
impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::Transport(e)
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
