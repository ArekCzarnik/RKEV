use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::types::{SystemOneRequest, SystemOneResponse};

/// Default address of a locally served Kev
/// (`python -m kev.serve --run jaredpalmer/kev-4b --port 8009`).
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8009";

/// Model alias the server resolves to whatever checkpoint it loaded.
pub const DEFAULT_MODEL: &str = "kev-latest";

const REQUEST_ID_HEADER: &str = "x-typesafe-request-id";

/// An HTTP client for a Kev server.
///
/// ```no_run
/// # async fn run() -> Result<(), kev_client::Error> {
/// use kev_client::{Client, Noul, SystemOneRequest};
///
/// let client = Client::local()?;
/// let response = client
///     .system_one(&SystemOneRequest::new("I was charged twice.")
///         .ask("billing", Noul::new("Is this ticket about billing?")))
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
}

impl Client {
    /// A client for a Kev server at `base_url`, e.g. `http://127.0.0.1:8009`.
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(Error::InvalidBaseUrl(base_url));
        }
        Ok(Self {
            http: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            model: DEFAULT_MODEL.to_string(),
        })
    }

    /// A client for a Kev server on the default local port.
    pub fn local() -> Result<Self> {
        Self::new(DEFAULT_BASE_URL)
    }

    /// Send `Authorization: Bearer <key>`. Required when the server runs with
    /// `KEV_API_KEY` set; harmless otherwise.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// The model sent with requests that do not name one themselves.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Use your own `reqwest::Client`, for timeouts, proxies or connection reuse.
    pub fn with_http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// Ask all questions in one forward pass — the normal call.
    pub async fn system_one(&self, request: &SystemOneRequest) -> Result<SystemOneResponse> {
        self.post_system_one("/v1/systemone", request).await
    }

    /// Ask each question in its own forward pass.
    ///
    /// Questions are already isolated from each other in the normal call; this
    /// endpoint exists to verify that, and costs one pass per question.
    pub async fn system_one_separate(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        self.post_system_one("/v1/systemone/separate", request)
            .await
    }

    /// Run one `choice` question under several option orders, to see whether
    /// the order moves the answer.
    ///
    /// `n_perm` is clamped to the server's range of 1 to 64. The response shape
    /// of this endpoint is not part of the documented API, so it comes back as
    /// raw JSON.
    pub async fn permute(&self, request: &SystemOneRequest, n_perm: u8) -> Result<Value> {
        let mut body = serde_json::to_value(self.with_model_filled_in(request))
            .map_err(|source| Error::Decode {
                source,
                body: String::from("the request could not be serialised"),
            })?;
        if let Some(object) = body.as_object_mut() {
            object.insert("n_perm".into(), Value::from(n_perm.clamp(1, 64)));
        }
        let (_, value) = self.post_json("/v1/systemone/permute", &body).await?;
        Ok(value)
    }

    /// The server's model cards plus details of the loaded checkpoint.
    ///
    /// Returned as raw JSON: the envelope is not pinned down by the API docs,
    /// so a typed struct here would only invent a contract.
    pub async fn models(&self) -> Result<Value> {
        let url = format!("{}/v1/models", self.base_url);
        let mut builder = self.http.get(url);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let (_, value) = Self::read(builder).await?;
        Ok(value)
    }

    fn with_model_filled_in(&self, request: &SystemOneRequest) -> SystemOneRequest {
        let mut request = request.clone();
        if request.model.is_none() {
            request.model = Some(self.model.clone());
        }
        request
    }

    async fn post_system_one(
        &self,
        path: &str,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse> {
        let body = self.with_model_filled_in(request);
        let (request_id, mut response): (Option<String>, SystemOneResponse) =
            self.post_json(path, &body).await?;
        response.request_id = request_id;
        Ok(response)
    }

    async fn post_json<T: DeserializeOwned, B: serde::Serialize + ?Sized>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<(Option<String>, T)> {
        let url = format!("{}{}", self.base_url, path);
        let mut builder = self.http.post(url).json(body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        Self::read(builder).await
    }

    async fn read<T: DeserializeOwned>(
        builder: reqwest::RequestBuilder,
    ) -> Result<(Option<String>, T)> {
        let response = builder.send().await?;
        let status = response.status();
        let request_id = response
            .headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::Api {
                status: status.as_u16(),
                request_id,
                body,
            });
        }

        let parsed =
            serde_json::from_str::<T>(&body).map_err(|source| Error::Decode { source, body })?;
        Ok((request_id, parsed))
    }
}
