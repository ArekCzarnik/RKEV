//! The candle Qwen3.5 backbone: attention mixed with Gated DeltaNet.
//!
//! Same approach as `tests/qwen3.rs` — a checkpoint written here, made of noise,
//! and a second implementation to compare against. The second implementation
//! matters more on this architecture than on the last one: a gated delta rule
//! has a decay, a write strength, an L2 norm, a short convolution and a gated
//! output norm, and every one of them is a place to be quietly wrong.
//!
//! Transcribed from `transformers/models/qwen3_5/modeling_qwen3_5.py`.

#![cfg(feature = "candle")]

mod fixtures;

use std::fs;
use std::path::PathBuf;

use fixtures::{
    fresh_dir, tokenizer_json, vocab_size, write_head, write_safetensors, Noise, Tensors,
};
use kev_client::{pointer_head, Backend, Forward, LocalEngine, Noul, Pass, SystemOneRequest};

const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 64;
const LAYERS: usize = 2;
/// The first layer is the recurrence, the second is attention.
const LAYER_TYPES: [&str; LAYERS] = ["linear_attention", "full_attention"];
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
/// Hugging Face's default `partial_rotary_factor` is 0.25, which the config
/// below leaves unstated on purpose.
const ROTARY: usize = HEAD_DIM / 4;
const KEY_HEADS: usize = 2;
const VALUE_HEADS: usize = 4;
const KEY_DIM: usize = 4;
const VALUE_DIM: usize = 6;
const CONV_KERNEL: usize = 3;
const POINTER_DIM: usize = 16;
const EPS: f32 = 1e-6;
const THETA: f64 = 10_000.0;

const KEYS: usize = KEY_HEADS * KEY_DIM;
const VALUES: usize = VALUE_HEADS * VALUE_DIM;
const CONV_DIM: usize = 2 * KEYS + VALUES;

struct Fixture {
    dir: PathBuf,
    tensors: Tensors,
}

/// A hybrid checkpoint. With `adapter`, a LoRA on the recurrence's qkv
/// projection is shipped alongside; with `merge`, the same delta is folded into
/// the weight instead.
fn checkpoint(name: &str, adapter: bool, merge: bool) -> Fixture {
    let dir = fresh_dir(&format!("qwen3_5-{name}"));
    fs::write(
        dir.join("config.json"),
        format!(
            r#"{{"hidden_size":{HIDDEN},"intermediate_size":{INTERMEDIATE},
                "num_hidden_layers":{LAYERS},"num_attention_heads":{HEADS},
                "num_key_value_heads":{KV_HEADS},"head_dim":{HEAD_DIM},
                "layer_types":["{}","{}"],
                "linear_num_key_heads":{KEY_HEADS},"linear_num_value_heads":{VALUE_HEADS},
                "linear_key_head_dim":{KEY_DIM},"linear_value_head_dim":{VALUE_DIM},
                "linear_conv_kernel_dim":{CONV_KERNEL},
                "rms_norm_eps":1e-06,"rope_parameters":{{"rope_type":"default","rope_theta":{THETA}}},
                "vocab_size":{}}}"#,
            LAYER_TYPES[0],
            LAYER_TYPES[1],
            vocab_size()
        ),
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();

    let mut noise = Noise(11);
    let mut tensors = Tensors::new();
    let mut put = |name: String, shape: Vec<usize>, values: Vec<f32>| {
        tensors.insert(name, (shape, values));
    };
    put(
        String::from("model.embed_tokens.weight"),
        vec![vocab_size(), HIDDEN],
        noise.values(vocab_size() * HIDDEN),
    );
    for (layer, kind) in LAYER_TYPES.iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        // Every norm in this model is zero-centred: the weight is the deviation
        // from 1.0, so noise around 0 is what a trained one looks like.
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            put(
                format!("{prefix}.{norm}.weight"),
                vec![HIDDEN],
                noise.values(HIDDEN),
            );
        }
        if *kind == "linear_attention" {
            let linear = format!("{prefix}.linear_attn");
            put(
                format!("{linear}.in_proj_qkv.weight"),
                vec![CONV_DIM, HIDDEN],
                noise.values(CONV_DIM * HIDDEN),
            );
            put(
                format!("{linear}.in_proj_z.weight"),
                vec![VALUES, HIDDEN],
                noise.values(VALUES * HIDDEN),
            );
            for name in ["in_proj_b", "in_proj_a"] {
                put(
                    format!("{linear}.{name}.weight"),
                    vec![VALUE_HEADS, HIDDEN],
                    noise.values(VALUE_HEADS * HIDDEN),
                );
            }
            put(
                format!("{linear}.out_proj.weight"),
                vec![HIDDEN, VALUES],
                noise.values(HIDDEN * VALUES),
            );
            put(
                format!("{linear}.conv1d.weight"),
                vec![CONV_DIM, 1, CONV_KERNEL],
                noise.values(CONV_DIM * CONV_KERNEL),
            );
            // dt_bias starts at ones and A_log at log(A) for A in (0.01, 16).
            put(
                format!("{linear}.dt_bias"),
                vec![VALUE_HEADS],
                noise.around_one(VALUE_HEADS),
            );
            put(
                format!("{linear}.A_log"),
                vec![VALUE_HEADS],
                noise.values(VALUE_HEADS),
            );
            // The gated output norm is the one norm here that is ones-centred.
            put(
                format!("{linear}.norm.weight"),
                vec![VALUE_DIM],
                noise.around_one(VALUE_DIM),
            );
        } else {
            let attention = format!("{prefix}.self_attn");
            // Twice as wide as Qwen3's: half of every head is an output gate.
            put(
                format!("{attention}.q_proj.weight"),
                vec![HEADS * HEAD_DIM * 2, HIDDEN],
                noise.values(HEADS * HEAD_DIM * 2 * HIDDEN),
            );
            for name in ["k_proj", "v_proj"] {
                put(
                    format!("{attention}.{name}.weight"),
                    vec![KV_HEADS * HEAD_DIM, HIDDEN],
                    noise.values(KV_HEADS * HEAD_DIM * HIDDEN),
                );
            }
            put(
                format!("{attention}.o_proj.weight"),
                vec![HIDDEN, HEADS * HEAD_DIM],
                noise.values(HIDDEN * HEADS * HEAD_DIM),
            );
            for name in ["q_norm", "k_norm"] {
                put(
                    format!("{attention}.{name}.weight"),
                    vec![HEAD_DIM],
                    noise.values(HEAD_DIM),
                );
            }
        }
        for name in ["mlp.gate_proj", "mlp.up_proj"] {
            put(
                format!("{prefix}.{name}.weight"),
                vec![INTERMEDIATE, HIDDEN],
                noise.values(INTERMEDIATE * HIDDEN),
            );
        }
        put(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
            noise.values(HIDDEN * INTERMEDIATE),
        );
    }
    put(
        String::from("model.norm.weight"),
        vec![HIDDEN],
        noise.values(HIDDEN),
    );

    if adapter || merge {
        let rank = 2;
        let mut lora = Noise(3);
        let a = lora.values(rank * HIDDEN);
        let b = lora.values(CONV_DIM * rank);
        let scale = 4.0 / rank as f32;
        if merge {
            let target = tensors
                .get_mut("model.layers.0.linear_attn.in_proj_qkv.weight")
                .unwrap();
            for row in 0..CONV_DIM {
                for column in 0..HIDDEN {
                    let delta: f32 = (0..rank)
                        .map(|r| b[row * rank + r] * a[r * HIDDEN + column])
                        .sum();
                    target.1[row * HIDDEN + column] += delta * scale;
                }
            }
        } else {
            let mut adapter_tensors = Tensors::new();
            adapter_tensors.insert(
                String::from("base_model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight"),
                (vec![rank, HIDDEN], a),
            );
            adapter_tensors.insert(
                String::from("base_model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"),
                (vec![CONV_DIM, rank], b),
            );
            let adapter_dir = dir.join("adapter");
            fs::create_dir_all(&adapter_dir).unwrap();
            write_safetensors(
                &adapter_dir.join("adapter_model.safetensors"),
                &adapter_tensors,
            );
            fs::write(
                adapter_dir.join("adapter_config.json"),
                r#"{"r":2,"lora_alpha":4,"target_modules":["in_proj_qkv"]}"#,
            )
            .unwrap();
        }
    }

    write_safetensors(&dir.join("model.safetensors"), &tensors);
    write_head(&dir, HIDDEN, POINTER_DIM);
    Fixture { dir, tensors }
}

fn engine(fixture: &Fixture, adapter: Option<PathBuf>) -> LocalEngine {
    let backend = Backend::open(&fixture.dir, adapter.as_deref()).unwrap();
    LocalEngine::new(
        backend,
        pointer_head(&fixture.dir.join("head.safetensors")).unwrap(),
    )
}

fn a_request() -> SystemOneRequest {
    SystemOneRequest::new("a ticket about money")
        .ask("money", Noul::new("is this about money ?"))
        .ask("late", Noul::new("is this late ?"))
}

fn probability(response: &kev_client::SystemOneResponse, id: &str) -> f64 {
    response.answer(id).unwrap().as_noul().unwrap()
}

// ---------------------------------------------------------------------------
// A second implementation, to check the first one's conventions
// ---------------------------------------------------------------------------

fn weights<'a>(tensors: &'a Tensors, name: &str) -> &'a [f32] {
    &tensors
        .get(name)
        .unwrap_or_else(|| panic!("no tensor {name}"))
        .1
}

fn matvec(weight: &[f32], outputs: usize, inputs: usize, x: &[f32]) -> Vec<f32> {
    (0..outputs)
        .map(|row| {
            (0..inputs)
                .map(|column| weight[row * inputs + column] * x[column])
                .sum()
        })
        .collect()
}

/// `Qwen3_5RMSNorm`: zero-centred, so the weight is the deviation from 1.0.
fn rms_norm(x: &[f32], weight: &[f32]) -> Vec<f32> {
    let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    x.iter()
        .zip(weight)
        .map(|(x, w)| x * scale * (1.0 + w))
        .collect()
}

/// `Qwen3_5RMSNormGated`: ones-centred, and multiplied by `silu(gate)` after the
/// norm rather than before it.
fn rms_norm_gated(x: &[f32], weight: &[f32], gate: &[f32]) -> Vec<f32> {
    let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + EPS).sqrt();
    x.iter()
        .zip(weight)
        .zip(gate)
        .map(|((x, w), gate)| x * scale * w * silu(*gate))
        .collect()
}

/// `l2norm` from the FLA library, which is not an RMS norm: no division by the
/// width.
fn l2_norm(x: &[f32]) -> Vec<f32> {
    let scale = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + EPS).sqrt();
    x.iter().map(|v| v * scale).collect()
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Partial rotary: only the first `ROTARY` components of a head are rotated, and
/// the rest pass through (`apply_rotary_pos_emb` slices at `cos.shape[-1]`).
fn rotate(head: &mut [f32], position: u32) {
    for i in 0..ROTARY / 2 {
        let angle = position as f64 * THETA.powf(-2.0 * i as f64 / ROTARY as f64);
        let (cos, sin) = (angle.cos() as f32, angle.sin() as f32);
        let (low, high) = (head[i], head[i + ROTARY / 2]);
        head[i] = low * cos - high * sin;
        head[i + ROTARY / 2] = high * cos + low * sin;
    }
}

/// `Qwen3_5TextModel.forward` in plain f32: the recurrence, the gated attention,
/// and the zero-centred norms, transcribed from the reference.
fn reference_hidden(
    tensors: &Tensors,
    ids: &[u32],
    positions: &[u32],
    attends: &dyn Fn(usize, usize) -> bool,
) -> Vec<Vec<f32>> {
    let embed = weights(tensors, "model.embed_tokens.weight");
    let mut xs: Vec<Vec<f32>> = ids
        .iter()
        .map(|id| embed[*id as usize * HIDDEN..(*id as usize + 1) * HIDDEN].to_vec())
        .collect();

    for (layer, kind) in LAYER_TYPES.iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        let of = |name: &str| weights(tensors, &format!("{prefix}.{name}"));
        let normed: Vec<Vec<f32>> = xs
            .iter()
            .map(|x| rms_norm(x, of("input_layernorm.weight")))
            .collect();

        let mixed = if *kind == "linear_attention" {
            recurrence(&normed, &|name| of(&format!("linear_attn.{name}")))
        } else {
            attention(&normed, positions, attends, &|name| {
                of(&format!("self_attn.{name}"))
            })
        };
        for (x, mixed) in xs.iter_mut().zip(mixed) {
            for (x, mixed) in x.iter_mut().zip(mixed) {
                *x += mixed;
            }
        }

        for x in xs.iter_mut() {
            let normed = rms_norm(x, of("post_attention_layernorm.weight"));
            let gate = matvec(of("mlp.gate_proj.weight"), INTERMEDIATE, HIDDEN, &normed);
            let up = matvec(of("mlp.up_proj.weight"), INTERMEDIATE, HIDDEN, &normed);
            let activated: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let down = matvec(of("mlp.down_proj.weight"), HIDDEN, INTERMEDIATE, &activated);
            for (x, down) in x.iter_mut().zip(down) {
                *x += down;
            }
        }
    }

    xs.iter()
        .map(|x| rms_norm(x, weights(tensors, "model.norm.weight")))
        .collect()
}

/// `Qwen3_5GatedDeltaNet.forward` plus `torch_recurrent_gated_delta_rule`.
fn recurrence<'a>(normed: &[Vec<f32>], of: &dyn Fn(&str) -> &'a [f32]) -> Vec<Vec<f32>> {
    let len = normed.len();

    // The shared projection, then the depthwise causal convolution over time:
    // out[t] = sum_j w[j] * x[t - (K - 1) + j], which is what F.conv1d with
    // padding K-1 gives once the tail is dropped.
    let projected: Vec<Vec<f32>> = normed
        .iter()
        .map(|x| matvec(of("in_proj_qkv.weight"), CONV_DIM, HIDDEN, x))
        .collect();
    let conv = of("conv1d.weight");
    let mixed: Vec<Vec<f32>> = (0..len)
        .map(|t| {
            (0..CONV_DIM)
                .map(|channel| {
                    let sum: f32 = (0..CONV_KERNEL)
                        .filter_map(|j| {
                            let step = (t + j).checked_sub(CONV_KERNEL - 1)?;
                            Some(conv[channel * CONV_KERNEL + j] * projected[step][channel])
                        })
                        .sum();
                    silu(sum)
                })
                .collect()
        })
        .collect();

    // One state per value head, [key dim][value dim], and the key heads are
    // shared: value head h reads key head h / (value heads / key heads).
    let group = VALUE_HEADS / KEY_HEADS;
    let mut state = vec![vec![vec![0.0f32; VALUE_DIM]; KEY_DIM]; VALUE_HEADS];
    let mut out = Vec::with_capacity(len);

    for (t, mixed) in mixed.iter().enumerate() {
        let beta: Vec<f32> = matvec(of("in_proj_b.weight"), VALUE_HEADS, HIDDEN, &normed[t])
            .into_iter()
            .map(sigmoid)
            .collect();
        let a = matvec(of("in_proj_a.weight"), VALUE_HEADS, HIDDEN, &normed[t]);
        let z = matvec(of("in_proj_z.weight"), VALUES, HIDDEN, &normed[t]);
        let (dt_bias, a_log) = (of("dt_bias"), of("A_log"));

        let mut attended = Vec::with_capacity(VALUES);
        for head in 0..VALUE_HEADS {
            let key_head = head / group;
            let q = l2_norm(&mixed[key_head * KEY_DIM..(key_head + 1) * KEY_DIM]);
            let k = l2_norm(&mixed[KEYS + key_head * KEY_DIM..KEYS + (key_head + 1) * KEY_DIM]);
            // The query is scaled by the key width, not the value width.
            let q: Vec<f32> = q.iter().map(|v| v / (KEY_DIM as f32).sqrt()).collect();
            let v = &mixed[2 * KEYS + head * VALUE_DIM..2 * KEYS + (head + 1) * VALUE_DIM];

            let decay = (-a_log[head].exp() * softplus(a[head] + dt_bias[head])).exp();
            let state = &mut state[head];
            for row in state.iter_mut() {
                for value in row.iter_mut() {
                    *value *= decay;
                }
            }
            // How much the memory is off by, scaled by this token's write
            // strength, written back as an outer product.
            let remembered: Vec<f32> = (0..VALUE_DIM)
                .map(|column| (0..KEY_DIM).map(|row| state[row][column] * k[row]).sum())
                .collect();
            let delta: Vec<f32> = (0..VALUE_DIM)
                .map(|column| (v[column] - remembered[column]) * beta[head])
                .collect();
            for row in 0..KEY_DIM {
                for column in 0..VALUE_DIM {
                    state[row][column] += k[row] * delta[column];
                }
            }
            let read: Vec<f32> = (0..VALUE_DIM)
                .map(|column| (0..KEY_DIM).map(|row| state[row][column] * q[row]).sum())
                .collect();
            attended.extend(rms_norm_gated(
                &read,
                of("norm.weight"),
                &z[head * VALUE_DIM..(head + 1) * VALUE_DIM],
            ));
        }
        out.push(matvec(of("out_proj.weight"), HIDDEN, VALUES, &attended));
    }
    out
}

/// `Qwen3_5Attention.forward`: Qwen3's attention, with half of `q_proj` taken as
/// an output gate and only a quarter of every head rotated.
fn attention<'a>(
    normed: &[Vec<f32>],
    positions: &[u32],
    attends: &dyn Fn(usize, usize) -> bool,
    of: &dyn Fn(&str) -> &'a [f32],
) -> Vec<Vec<f32>> {
    let len = normed.len();
    let groups = HEADS / KV_HEADS;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut queries = Vec::new();
    let mut gates = Vec::new();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for (token, x) in normed.iter().enumerate() {
        let projected = matvec(of("q_proj.weight"), HEADS * HEAD_DIM * 2, HIDDEN, x);
        let mut q = Vec::new();
        let mut gate = Vec::new();
        for head in 0..HEADS {
            let at = head * HEAD_DIM * 2;
            let mut rotated = rms_norm(&projected[at..at + HEAD_DIM], of("q_norm.weight"));
            rotate(&mut rotated, positions[token]);
            q.push(rotated);
            gate.extend(&projected[at + HEAD_DIM..at + 2 * HEAD_DIM]);
        }
        queries.push(q);
        gates.push(gate);

        let k = matvec(of("k_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, x);
        let v = matvec(of("v_proj.weight"), KV_HEADS * HEAD_DIM, HIDDEN, x);
        keys.push(
            k.chunks(HEAD_DIM)
                .map(|head| {
                    let mut head = rms_norm(head, of("k_norm.weight"));
                    rotate(&mut head, positions[token]);
                    head
                })
                .collect::<Vec<_>>(),
        );
        values.push(v.chunks(HEAD_DIM).map(<[f32]>::to_vec).collect::<Vec<_>>());
    }

    (0..len)
        .map(|token| {
            let mut attended = vec![0.0; HEADS * HEAD_DIM];
            for head in 0..HEADS {
                let kv = head / groups;
                let visible: Vec<usize> = (0..len).filter(|key| attends(token, *key)).collect();
                let scores: Vec<f32> = visible
                    .iter()
                    .map(|key| {
                        let dot: f32 = queries[token][head]
                            .iter()
                            .zip(&keys[*key][kv])
                            .map(|(q, k)| q * k)
                            .sum();
                        dot * scale
                    })
                    .collect();
                let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exponentials: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                let total: f32 = exponentials.iter().sum();
                for (key, weight) in visible.iter().zip(&exponentials) {
                    for i in 0..HEAD_DIM {
                        attended[head * HEAD_DIM + i] += weight / total * values[*key][kv][i];
                    }
                }
            }
            // The gate closes after the attention, before the projection out.
            let gated: Vec<f32> = attended
                .iter()
                .zip(&gates[token])
                .map(|(value, gate)| value * sigmoid(*gate))
                .collect();
            matvec(of("o_proj.weight"), HIDDEN, HEADS * HEAD_DIM, &gated)
        })
        .collect()
}

// ---------------------------------------------------------------------------

#[test]
fn a_hybrid_config_gets_the_hybrid_backbone() {
    let fixture = checkpoint("dispatch", false, false);

    let backend = Backend::open(&fixture.dir, None).unwrap();

    assert!(backend.is_hybrid(), "the recurrence was not noticed");
    assert_eq!(backend.hidden_size(), HIDDEN);
}

#[test]
fn the_forward_pass_matches_a_transcription_of_modeling_qwen3_5() {
    // One row, since that is what a hybrid base ever runs: the state followed by
    // one question's branch. Every hidden unit of every token is compared, not
    // just the ones the readout looks at.
    let fixture = checkpoint("reference", false, false);
    let mut backend = Backend::open(&fixture.dir, None).unwrap();

    let ids: Vec<u32> = vec![1, 6, 7, 2, 12, 3, 14, 4, 5];
    let segments: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 1, 1, 1];
    let positions: Vec<u32> = (0..ids.len() as u32).collect();
    // Every position of the question's branch: `Pass::rows` returns readout
    // positions inside a branch, not ones in the state, and the readout only
    // ever points at a branch anyway.
    let readout: Vec<usize> = (3..ids.len()).collect();
    let pass = Pass {
        ids: &ids,
        positions: &positions,
        segments: &segments,
        readout: &readout,
    };

    let ours = backend.hidden(&pass).unwrap();
    let reference: Vec<Vec<f32>> =
        reference_hidden(&fixture.tensors, &ids, &positions, &|query, key| {
            pass.attends(query, key)
        })
        .into_iter()
        .skip(3)
        .collect();

    // They agree to about 1e-6, f32 accumulation order; 1e-5 leaves room for
    // another CPU without letting a wrong convention through.
    assert_eq!(ours.len(), reference.len());
    for (token, (ours, reference)) in ours.iter().zip(&reference).enumerate() {
        for (i, (ours, reference)) in ours.iter().zip(reference).enumerate() {
            assert!(
                (ours - reference).abs() <= 1e-5,
                "token {token}, hidden unit {i}: {ours} vs {reference}"
            );
        }
    }
}

#[test]
fn every_question_gets_its_own_row_so_the_isolation_is_exact() {
    // A recurrence cannot be masked, so questions are kept apart by running one
    // row each. That makes asking together and asking separately the same
    // computation, not merely a close one.
    let fixture = checkpoint("rows", false, false);
    let engine = engine(&fixture, None);

    let together = engine.system_one_blocking(&a_request()).unwrap();
    let apart = engine.system_one_separate_blocking(&a_request()).unwrap();
    let alone = engine
        .system_one_blocking(
            &SystemOneRequest::new("a ticket about money")
                .ask("money", Noul::new("is this about money ?")),
        )
        .unwrap();

    for id in ["money", "late"] {
        assert_eq!(
            probability(&together, id),
            probability(&apart, id),
            "{id} differs between the packed and the separate call"
        );
    }
    assert_eq!(
        probability(&together, "money"),
        probability(&alone, "money"),
        "a second question changed the first one's answer"
    );
}

#[test]
fn the_engine_answers_over_the_hybrid_backbone() {
    let fixture = checkpoint("answers", false, false);

    let response = engine(&fixture, None)
        .system_one_blocking(&a_request())
        .unwrap();

    for id in ["money", "late"] {
        let probability = probability(&response, id);
        assert!(
            (0.0..=1.0).contains(&probability),
            "{id} answered {probability}"
        );
    }
    assert!(response.usage.input_tokens > 0);
}

#[test]
fn an_adapter_is_merged_into_the_recurrence() {
    // The adapter covers the DeltaNet projections too (kev.train picks them from
    // the config), so merging has to reach them.
    let with_adapter = checkpoint("adapter", true, false);
    let premerged = checkpoint("premerged", false, true);

    let adapted = engine(&with_adapter, Some(with_adapter.dir.join("adapter")))
        .system_one_blocking(&a_request())
        .unwrap();
    let merged = engine(&premerged, None)
        .system_one_blocking(&a_request())
        .unwrap();
    let base = engine(&with_adapter, None)
        .system_one_blocking(&a_request())
        .unwrap();

    assert!(
        (probability(&adapted, "money") - probability(&merged, "money")).abs() < 1e-4,
        "merging the adapter into the recurrence is not what peft does: {} vs {}",
        probability(&adapted, "money"),
        probability(&merged, "money")
    );
    assert_ne!(
        probability(&adapted, "money"),
        probability(&base, "money"),
        "the adapter changed nothing, so it was not applied"
    );
}
