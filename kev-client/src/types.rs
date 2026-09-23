//! Request and response types for `POST /v1/systemone`.
//!
//! The wire format is TypeSafe's System One format, which Kev implements.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

/// One System One call: a `state` to evaluate plus the questions to ask about
/// it. Questions share the state but cannot read each other.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    /// The content to evaluate. A string, or any JSON object/array — Kev
    /// converts objects and arrays to labelled text.
    pub state: Value,
    /// Left empty here, the [`Client`](crate::Client) fills in its own model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Question id -> question. The id is yours; the model never sees it.
    pub questions: IndexMap<String, Question>,
}

impl SystemOneRequest {
    /// Start a request for the given state.
    pub fn new(state: impl Into<Value>) -> Self {
        Self {
            state: state.into(),
            model: None,
            questions: IndexMap::new(),
        }
    }

    /// Start a request whose state is any serialisable value (a struct, a map,
    /// a list of messages, ...).
    pub fn from_state<T: Serialize>(state: &T) -> serde_json::Result<Self> {
        Ok(Self::new(serde_json::to_value(state)?))
    }

    /// Pin this request to a specific model, overriding the client default.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Add a question under the given id.
    pub fn ask(mut self, id: impl Into<String>, question: impl Into<Question>) -> Self {
        self.questions.insert(id.into(), question.into());
        self
    }
}

/// A single question. Build one with [`Noul`], [`Choice`] or [`Score`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no. Answered with the probability of yes.
    Noul(Noul),
    /// Pick one of 1–255 named options.
    Choice(Choice),
    /// Rate on an ordered scale of 1–255 levels.
    Score(Score),
}

/// A yes/no question.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Noul {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Optional descriptions of what yes and no mean.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

impl Noul {
    /// A yes/no question with the given instructions.
    pub fn new(instructions: impl Into<Value>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: None,
        }
    }

    /// Describe what a yes means.
    pub fn yes(mut self, description: impl Into<String>) -> Self {
        self.criteria.get_or_insert_with(Default::default).yes = Some(description.into());
        self
    }

    /// Describe what a no means.
    pub fn no(mut self, description: impl Into<String>) -> Self {
        self.criteria.get_or_insert_with(Default::default).no = Some(description.into());
        self
    }
}

/// A multiple-choice question.
///
/// Option order is part of the request and can change the answer, so the
/// options keep the order you add them in.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Choice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    pub criteria: IndexMap<String, Option<String>>,
}

impl Choice {
    /// A multiple-choice question with the given instructions.
    pub fn new(instructions: impl Into<Value>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: IndexMap::new(),
        }
    }

    /// Add an option together with a description of when it applies.
    pub fn option(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.criteria.insert(name.into(), Some(description.into()));
        self
    }

    /// Add an option whose name has to speak for itself (serialised as `null`).
    pub fn option_bare(mut self, name: impl Into<String>) -> Self {
        self.criteria.insert(name.into(), None);
        self
    }

    /// Add several described options at once, keeping their order.
    pub fn options<K, V, I>(mut self, options: I) -> Self
    where
        K: Into<String>,
        V: Into<String>,
        I: IntoIterator<Item = (K, V)>,
    {
        for (name, description) in options {
            self.criteria.insert(name.into(), Some(description.into()));
        }
        self
    }
}

/// A rating question over an ordered scale.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Score {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>,
    pub criteria: Vec<String>,
}

impl Score {
    /// A rating question with the given instructions.
    pub fn new(instructions: impl Into<Value>) -> Self {
        Self {
            instructions: Some(instructions.into()),
            criteria: Vec::new(),
        }
    }

    /// Append the next level. Add them from lowest to highest.
    pub fn level(mut self, description: impl Into<String>) -> Self {
        self.criteria.push(description.into());
        self
    }

    /// Append several levels, lowest first.
    pub fn levels<S, I>(mut self, levels: I) -> Self
    where
        S: Into<String>,
        I: IntoIterator<Item = S>,
    {
        self.criteria.extend(levels.into_iter().map(Into::into));
        self
    }
}

impl From<Noul> for Question {
    fn from(q: Noul) -> Self {
        Question::Noul(q)
    }
}

impl From<Choice> for Question {
    fn from(q: Choice) -> Self {
        Question::Choice(q)
    }
}

impl From<Score> for Question {
    fn from(q: Score) -> Self {
        Question::Score(q)
    }
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

/// The answer set for one System One call.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneResponse {
    pub model: String,
    /// Same ids as the request, in the server's order.
    pub answers: IndexMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub latency_ms: Option<f64>,
    /// Taken from the `x-typesafe-request-id` header, not the body.
    #[serde(skip)]
    pub request_id: Option<String>,
}

impl SystemOneResponse {
    /// Look up one answer by the id you used in the request.
    pub fn answer(&self, id: &str) -> Option<&Answer> {
        self.answers.get(id)
    }
}

/// Token accounting. `output_tokens` counts the serialised answers, not
/// generated tokens — Kev generates none.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

/// One answer, shaped by the question type it belongs to.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// Probability of yes, 0.0 to 1.0.
        noul: f64,
    },
    Choice {
        /// The most likely option name.
        choice: String,
        #[serde(default)]
        confidence: f64,
        /// Option name -> probability, in the order the server returned.
        #[serde(default)]
        probabilities: IndexMap<String, f64>,
    },
    Score {
        /// Mean level index, starting at 0.
        score: f64,
        #[serde(default)]
        confidence: f64,
        /// Level index (as a string) -> the description you sent.
        #[serde(default)]
        legend: IndexMap<String, String>,
        /// Level index (as a string) -> probability.
        #[serde(default)]
        probabilities: IndexMap<String, f64>,
    },
}

impl Answer {
    /// Probability of yes, for a `noul` answer.
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Answer::Noul { noul } => Some(*noul),
            _ => None,
        }
    }

    /// The chosen option, for a `choice` answer.
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Answer::Choice { choice, .. } => Some(choice.as_str()),
            _ => None,
        }
    }

    /// The mean level index, for a `score` answer.
    pub fn as_score(&self) -> Option<f64> {
        match self {
            Answer::Score { score, .. } => Some(*score),
            _ => None,
        }
    }

    /// Confidence, where the server reports one. `noul` answers have none —
    /// the probability itself is the confidence there.
    ///
    /// Confidence is a shape measure of the distribution, not a measured
    /// accuracy rate.
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Answer::Noul { .. } => None,
            Answer::Choice { confidence, .. } | Answer::Score { confidence, .. } => {
                Some(*confidence)
            }
        }
    }

    /// The full distribution, where the server reports one.
    pub fn probabilities(&self) -> Option<&IndexMap<String, f64>> {
        match self {
            Answer::Noul { .. } => None,
            Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
                Some(probabilities)
            }
        }
    }

    /// The highest-probability label and its probability.
    ///
    /// For `score` the label is the level index as a string; pair it with
    /// [`Answer::legend`] for the description.
    pub fn top(&self) -> Option<(&str, f64)> {
        self.probabilities()?
            .iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(name, p)| (name.as_str(), *p))
    }

    /// The level descriptions, for a `score` answer.
    pub fn legend(&self) -> Option<&IndexMap<String, String>> {
        match self {
            Answer::Score { legend, .. } => Some(legend),
            _ => None,
        }
    }
}
