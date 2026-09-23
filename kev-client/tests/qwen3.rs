//! The candle Qwen3 backbone, over a checkpoint this test writes itself.
//!
//! Two layers, 32 hidden units and made-up weights: the numbers mean nothing,
//! which is the point. What is under test is everything that has to hold for
//! *any* weights — that a question cannot read another question, that the
//! packed pass and the row form agree, that an adapter is merged the way peft
//! merges it — and those are exactly the properties the Python checks
//! (`tests/test_model.py::test_rows_match_packed`).
//!
//! Real weights can only be checked against a running Kev server; see
//! `.claude/tasks/2026-09-23-local-inference.md`.

#![cfg(feature = "qwen3")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use kev_client::{
    pointer_head, Choice, Forward, LocalEngine, Noul, Pass, Qwen3Backend, SystemOneRequest, DECIDE,
    OPTION, OPTION_END, QUESTION, STATE,
};

const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 64;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const POINTER_DIM: usize = 16;

/// Words the toy tokenizer knows; everything else becomes `[UNK]`.
const WORDS: [&str; 16] = [
    "a", "ticket", "about", "money", "late", "shoes", "no", "yes", "returns", "billing", "calm",
    "angry", "is", "this", "?", "the",
];

// ---------------------------------------------------------------------------
// A checkpoint on disk
// ---------------------------------------------------------------------------

/// Deterministic small values, so a test failure is always the code's fault.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 0.2
    }

    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// The minimum a safetensors reader needs: a u64 header length, a JSON header,
/// then the f32s. Written by hand so the tests need no extra dependency.
fn write_safetensors(path: &Path, tensors: &BTreeMap<String, (Vec<usize>, Vec<f32>)>) {
    let mut header = String::from("{");
    let mut offset = 0;
    for (index, (name, (shape, values))) in tensors.iter().enumerate() {
        let end = offset + values.len() * 4;
        if index > 0 {
            header.push(',');
        }
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{offset},{end}]}}"
        ));
        offset = end;
    }
    header.push('}');
    while (8 + header.len()) % 8 != 0 {
        header.push(' ');
    }

    let mut bytes = Vec::new();
    bytes.extend((header.len() as u64).to_le_bytes());
    bytes.extend(header.as_bytes());
    for (_, values) in tensors.values() {
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
    }
    fs::write(path, bytes).unwrap();
}

fn tokenizer_json() -> String {
    let mut vocab = vec![(String::from("[UNK]"), 0)];
    for (index, token) in [STATE, QUESTION, OPTION, OPTION_END, DECIDE]
        .iter()
        .enumerate()
    {
        vocab.push((token.to_string(), index + 1));
    }
    for (index, word) in WORDS.iter().enumerate() {
        vocab.push((word.to_string(), index + 6));
    }
    let entries: Vec<String> = vocab
        .iter()
        .map(|(token, id)| format!("\"{token}\":{id}"))
        .collect();
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,
            "decoder":null,
            "model":{{"type":"WordLevel","vocab":{{{}}},"unk_token":"[UNK]"}}}}"#,
        entries.join(",")
    )
}

fn vocab_size() -> usize {
    1 + 5 + WORDS.len()
}

/// A checkpoint directory: config, weights, tokenizer, pointer head. With
/// `adapter`, the LoRA tensors are written alongside; with `merge`, the same
/// delta is folded into the base weights instead.
fn checkpoint(name: &str, adapter: bool, merge: bool) -> PathBuf {
    // One fixed directory per test, rebuilt on every run rather than piling up
    // in the temp directory.
    let dir = std::env::temp_dir().join(format!("kev-qwen3-{name}"));
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).unwrap();

    fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"hidden_size":{HIDDEN},"intermediate_size":{INTERMEDIATE},
                "num_hidden_layers":{LAYERS},"num_attention_heads":{HEADS},
                "num_key_value_heads":{KV_HEADS},"head_dim":{HEAD_DIM},
                "rms_norm_eps":1e-06,"rope_theta":10000.0,"vocab_size":{}}}"#,
            vocab_size()
        ),
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let mut noise = Noise(42);
    let mut tensors = BTreeMap::new();
    tensors.insert(
        String::from("model.embed_tokens.weight"),
        (
            vec![vocab_size(), HIDDEN],
            noise.values(vocab_size() * HIDDEN),
        ),
    );
    for layer in 0..LAYERS {
        let prefix = format!("model.layers.{layer}");
        // Norm weights sit around 1.0, as trained ones do.
        let ones = |noise: &mut Noise, n: usize| {
            noise
                .values(n)
                .into_iter()
                .map(|v| 1.0 + v)
                .collect::<Vec<_>>()
        };
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            (vec![HIDDEN], ones(&mut noise, HIDDEN)),
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            (vec![HIDDEN], ones(&mut noise, HIDDEN)),
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            (vec![HEAD_DIM], ones(&mut noise, HEAD_DIM)),
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            (vec![HEAD_DIM], ones(&mut noise, HEAD_DIM)),
        );
        for (name, out) in [
            ("self_attn.q_proj", HEADS * HEAD_DIM),
            ("self_attn.k_proj", KV_HEADS * HEAD_DIM),
            ("self_attn.v_proj", KV_HEADS * HEAD_DIM),
        ] {
            tensors.insert(
                format!("{prefix}.{name}.weight"),
                (vec![out, HIDDEN], noise.values(out * HIDDEN)),
            );
        }
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            (
                vec![HIDDEN, HEADS * HEAD_DIM],
                noise.values(HIDDEN * HEADS * HEAD_DIM),
            ),
        );
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            (
                vec![INTERMEDIATE, HIDDEN],
                noise.values(INTERMEDIATE * HIDDEN),
            ),
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            (
                vec![INTERMEDIATE, HIDDEN],
                noise.values(INTERMEDIATE * HIDDEN),
            ),
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            (
                vec![HIDDEN, INTERMEDIATE],
                noise.values(HIDDEN * INTERMEDIATE),
            ),
        );
    }
    tensors.insert(
        String::from("model.norm.weight"),
        (
            vec![HIDDEN],
            noise.values(HIDDEN).into_iter().map(|v| 1.0 + v).collect(),
        ),
    );

    // One rank-2 LoRA on the first layer's query projection, either shipped as
    // an adapter or already folded into the weight.
    if adapter || merge {
        let rank = 2;
        let mut lora = Noise(7);
        let a = lora.values(rank * HIDDEN);
        let b = lora.values(HEADS * HEAD_DIM * rank);
        let scale = 4.0 / rank as f32; // lora_alpha / r
        if merge {
            let target = tensors
                .get_mut("model.layers.0.self_attn.q_proj.weight")
                .unwrap();
            for row in 0..HEADS * HEAD_DIM {
                for column in 0..HIDDEN {
                    let delta: f32 = (0..rank)
                        .map(|r| b[row * rank + r] * a[r * HIDDEN + column])
                        .sum();
                    target.1[row * HIDDEN + column] += delta * scale;
                }
            }
        } else {
            let mut adapter_tensors = BTreeMap::new();
            adapter_tensors.insert(
                String::from("base_model.model.layers.0.self_attn.q_proj.lora_A.weight"),
                (vec![rank, HIDDEN], a),
            );
            adapter_tensors.insert(
                String::from("base_model.model.layers.0.self_attn.q_proj.lora_B.weight"),
                (vec![HEADS * HEAD_DIM, rank], b),
            );
            let adapter_dir = dir.join("adapter");
            fs::create_dir_all(&adapter_dir).unwrap();
            write_safetensors(
                &adapter_dir.join("adapter_model.safetensors"),
                &adapter_tensors,
            );
            fs::write(
                adapter_dir.join("adapter_config.json"),
                r#"{"r":2,"lora_alpha":4,"target_modules":["q_proj"]}"#,
            )
            .unwrap();
        }
    }

    write_safetensors(&dir.join("model.safetensors"), &tensors);

    let mut head = Noise(99);
    let mut head_tensors = BTreeMap::new();
    for name in ["q", "k"] {
        head_tensors.insert(
            format!("head.{name}.weight"),
            (vec![POINTER_DIM, HIDDEN], head.values(POINTER_DIM * HIDDEN)),
        );
        head_tensors.insert(
            format!("head.{name}.bias"),
            (vec![POINTER_DIM], head.values(POINTER_DIM)),
        );
    }
    write_safetensors(&dir.join("head.safetensors"), &head_tensors);

    dir
}

fn engine(dir: &Path, adapter: Option<&Path>) -> LocalEngine {
    let backend = Qwen3Backend::open(dir, adapter).unwrap();
    let head = pointer_head(&dir.join("head.safetensors"), 1.0).unwrap();
    LocalEngine::new(backend, head)
}

fn a_request(second_question_options: (&str, &str)) -> SystemOneRequest {
    SystemOneRequest::new("a ticket about money")
        .ask(
            "team",
            Choice::new("is this about money ?")
                .option_bare("returns")
                .option_bare("billing"),
        )
        .ask(
            "escalate",
            Noul::new("is this late ?")
                .no(second_question_options.0)
                .yes(second_question_options.1),
        )
}

fn probabilities(response: &kev_client::SystemOneResponse, id: &str) -> Vec<f64> {
    let answer = response.answer(id).unwrap();
    match answer.probabilities() {
        Some(distribution) => distribution.values().copied().collect(),
        // A yes/no answer has no distribution: the probability is the answer.
        None => vec![answer.as_noul().unwrap()],
    }
}

fn assert_close(left: &[f64], right: &[f64], tolerance: f64, what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: different lengths");
    for (a, b) in left.iter().zip(right) {
        assert!((a - b).abs() <= tolerance, "{what}: {left:?} vs {right:?}");
    }
}

// ---------------------------------------------------------------------------

#[test]
fn the_backbone_answers_a_request() {
    let dir = checkpoint("answers", false, false);
    let engine = engine(&dir, None);

    let response = engine
        .system_one_blocking(&a_request(("calm", "angry")))
        .unwrap();

    let team = response.answer("team").unwrap();
    assert!(["returns", "billing"].contains(&team.as_choice().unwrap()));
    let total: f64 = probabilities(&response, "team").iter().sum();
    assert!((total - 1.0).abs() < 1e-3, "probabilities sum to {total}");
    let escalate = response.answer("escalate").unwrap().as_noul().unwrap();
    assert!((0.0..=1.0).contains(&escalate), "p(yes) = {escalate}");
    assert!(response.usage.input_tokens > 0);
}

#[test]
fn a_question_is_not_moved_by_another_question() {
    // The block-causal mask, end to end: question 2 changes, question 1 does
    // not. This is what `/v1/systemone/separate` exists to check, and it holds
    // for any weights.
    let dir = checkpoint("isolation", false, false);
    let engine = engine(&dir, None);

    let one = engine
        .system_one_blocking(&a_request(("calm", "angry")))
        .unwrap();
    let other = engine
        .system_one_blocking(&a_request(("shoes late", "money money money")))
        .unwrap();

    assert_close(
        &probabilities(&one, "team"),
        &probabilities(&other, "team"),
        1e-6,
        "question 1 moved when question 2 changed",
    );
    assert_ne!(
        probabilities(&one, "escalate"),
        probabilities(&other, "escalate"),
        "question 2 did not notice its own options changing"
    );
}

#[test]
fn asking_together_and_asking_separately_agree() {
    let dir = checkpoint("separate", false, false);
    let engine = engine(&dir, None);
    let request = a_request(("calm", "angry"));

    let packed = engine.system_one_blocking(&request).unwrap();
    let apart = engine.system_one_separate_blocking(&request).unwrap();

    for id in ["team", "escalate"] {
        assert_close(
            &probabilities(&packed, id),
            &probabilities(&apart, id),
            1e-5,
            id,
        );
    }
}

#[test]
fn the_packed_pass_and_the_row_form_agree() {
    // What a backbone with recurrent layers is limited to has to give the same
    // hidden states as the packed pass on a backbone that can do both.
    let dir = checkpoint("rows", false, false);
    let mut backend = Qwen3Backend::open(&dir, None).unwrap();
    let ids: Vec<u32> = vec![1, 6, 7, 2, 12, 3, 14, 4, 5, 2, 13, 3, 11, 4, 5];
    let segments: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2];
    let positions: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8];
    let readout: Vec<usize> = vec![8, 7, 14, 13];
    let pass = Pass {
        ids: &ids,
        positions: &positions,
        segments: &segments,
        readout: &readout,
    };

    let packed = backend.hidden(&pass).unwrap();
    let rows: Vec<Vec<f32>> = pass
        .rows()
        .iter()
        .flat_map(|row| backend.hidden(&row.as_pass()).unwrap())
        .collect();

    assert_eq!(packed.len(), rows.len());
    for (packed, row) in packed.iter().zip(&rows) {
        for (a, b) in packed.iter().zip(row) {
            assert!(
                (a - b).abs() <= 1e-4,
                "packed and row hidden states differ: {a} vs {b}"
            );
        }
    }
}

#[test]
fn an_adapter_is_merged_the_way_peft_merges_it() {
    // A checkpoint that ships base weights plus a LoRA must answer exactly like
    // one whose weights already carry the same delta.
    let with_adapter = checkpoint("adapter", true, false);
    let premerged = checkpoint("premerged", false, true);

    let adapted = engine(&with_adapter, Some(&with_adapter.join("adapter")))
        .system_one_blocking(&a_request(("calm", "angry")))
        .unwrap();
    let merged = engine(&premerged, None)
        .system_one_blocking(&a_request(("calm", "angry")))
        .unwrap();
    let base = engine(&with_adapter, None)
        .system_one_blocking(&a_request(("calm", "angry")))
        .unwrap();

    assert_close(
        &probabilities(&adapted, "team"),
        &probabilities(&merged, "team"),
        1e-5,
        "merging the adapter is not what peft does",
    );
    assert_ne!(
        probabilities(&adapted, "team"),
        probabilities(&base, "team"),
        "the adapter changed nothing, so it was not applied"
    );
}

#[test]
fn a_hybrid_base_is_refused_rather_than_answered_wrongly() {
    let dir = checkpoint("hybrid", false, false);
    let config = fs::read_to_string(dir.join("config.json")).unwrap();
    fs::write(
        dir.join("config.json"),
        config.replace(
            "\"rms_norm_eps\"",
            "\"layer_types\":[\"full_attention\",\"linear_attention\"],\"rms_norm_eps\"",
        ),
    )
    .unwrap();

    let error = Qwen3Backend::open(&dir, None).unwrap_err();

    assert!(error.to_string().contains("attention-only"), "{error}");
}
