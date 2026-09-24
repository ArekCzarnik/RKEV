//! From a System One request to the text the model sees.
//!
//! Mirrors `kev/api.py` (`render`, `option_text`, `to_record`). Every string
//! built here goes into the prompt, so any difference from the Python side is a
//! difference in the answers: the comments record where each rule comes from.

use indexmap::IndexMap;
use serde_json::Value;

use crate::types::{Question, SystemOneRequest};

/// Flatten a state or an instruction block into the text the model sees.
///
/// Field names are kept as labels, nested objects and arrays are indented two
/// spaces per level. `null` renders as the empty string.
pub fn render(value: &Value) -> String {
    render_at(value, 0)
}

fn render_at(value: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match value {
        Value::Null => String::new(),
        // Python's `str(True)`, which is what the training data and the server
        // both put in front of the model.
        Value::Bool(true) => String::from("True"),
        Value::Bool(false) => String::from("False"),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| format!("{pad}- {}", render_at(item, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(fields) => fields
            .iter()
            .map(|(key, field)| match field {
                // A nested document goes on its own indented lines; a scalar
                // stays on the label's line and is rendered without padding.
                Value::Array(_) | Value::Object(_) => {
                    format!("{pad}{key}:\n{}", render_at(field, indent + 1))
                }
                _ => format!("{pad}{key}: {}", render_at(field, 0)),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// One option as the model reads it: the bare name, or `name: description`.
///
/// A missing or empty description leaves the name to speak for itself.
fn option_text(name: &str, description: Option<&str>) -> String {
    match description {
        None | Some("") => name.to_string(),
        Some(description) => format!("{name}: {description}"),
    }
}

/// What kind of question an answer has to be read back as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kind {
    Noul,
    Choice,
    Score,
}

/// One question as the encoder sees it: instructions, then options in order.
#[derive(Debug, Clone)]
pub(crate) struct Ask {
    pub instructions: String,
    pub options: Vec<String>,
}

/// A whole request as the encoder sees it.
#[derive(Debug, Clone)]
pub(crate) struct Record {
    pub state: String,
    pub questions: Vec<Ask>,
}

/// How one question's distribution maps back onto an [`Answer`](crate::Answer).
///
/// The model only ever produces a probability per option; which answer shape
/// that becomes is decided here, not by the model.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    pub id: String,
    pub kind: Kind,
    /// The keys the probabilities are reported under, in option order.
    pub keys: Vec<String>,
    /// Level index -> description, for `score` questions only.
    pub legend: Option<IndexMap<String, String>>,
}

/// Split a request into the text to encode and the plan to read the answers
/// back with. Question and option order is preserved throughout — it is part
/// of the prompt and can move the answer.
pub(crate) fn plan(request: &SystemOneRequest) -> (Record, Vec<Plan>) {
    let mut questions = Vec::with_capacity(request.questions.len());
    let mut plans = Vec::with_capacity(request.questions.len());

    for (id, question) in &request.questions {
        let (kind, options, keys, legend) = match question {
            // A yes/no question is the same pointer primitive with two
            // options; the answer is the probability of the second one.
            Question::Noul(noul) => {
                let criteria = noul.criteria.as_ref();
                (
                    Kind::Noul,
                    vec![
                        option_text("no", criteria.and_then(|c| c.no.as_deref())),
                        option_text("yes", criteria.and_then(|c| c.yes.as_deref())),
                    ],
                    vec![String::from("false"), String::from("true")],
                    None,
                )
            }
            Question::Choice(choice) => (
                Kind::Choice,
                choice
                    .criteria
                    .iter()
                    .map(|(name, description)| option_text(name, description.as_deref()))
                    .collect(),
                choice.criteria.keys().cloned().collect(),
                None,
            ),
            // A score's options are the level descriptions, and the keys are
            // the level indices the distribution is reported under.
            Question::Score(score) => {
                let keys: Vec<String> = (0..score.criteria.len()).map(|i| i.to_string()).collect();
                let legend = keys
                    .iter()
                    .cloned()
                    .zip(score.criteria.iter().cloned())
                    .collect();
                (Kind::Score, score.criteria.clone(), keys, Some(legend))
            }
        };

        questions.push(Ask {
            instructions: instructions_of(question),
            options,
        });
        plans.push(Plan {
            id: id.clone(),
            kind,
            keys,
            legend,
        });
    }

    (
        Record {
            state: render(&request.state),
            questions,
        },
        plans,
    )
}

fn instructions_of(question: &Question) -> String {
    let instructions = match question {
        Question::Noul(q) => q.instructions.as_ref(),
        Question::Choice(q) => q.instructions.as_ref(),
        Question::Score(q) => q.instructions.as_ref(),
    };
    instructions.map(render).unwrap_or_default()
}
