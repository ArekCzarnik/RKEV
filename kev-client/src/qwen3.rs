//! The Qwen3 backbone, in candle: token ids in, hidden states out.
//!
//! This is the attention-only generation of Kev's bases (`Qwen3-4B-Base` and
//! friends, the `@qwen3` checkpoints). Two things make it different from the
//! Qwen3 implementations that ship with candle and mistral.rs, and they are the
//! whole reason it is written out here:
//!
//! * it takes an **arbitrary additive attention mask**, because Kev's questions
//!   must not read each other, and **explicit position ids**, because every
//!   question's branch restarts just after the state;
//! * it stops at the last hidden state. There is no vocabulary head — Kev reads
//!   its answers off the hidden states with a pointer head, and never generates.
//!
//! Everything runs in f32, the precision the published numbers were measured
//! at, and there is no KV cache: one prefill pass answers a whole request.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::weights::{
    attention_mask, branch_batch_mask, linear, pad_rows, prefill_batch_mask, repeat_kv, rms_norm,
    rope, rotary_tables, Weights,
};

/// The parts of a Qwen3 `config.json` this backbone needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Qwen3 states it rather than deriving it from the hidden size.
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    /// Older configs put it at the top level, newer ones under
    /// `rope_parameters`; [`Config::rope_theta`] reads whichever is there.
    rope_theta: Option<f64>,
    rope_parameters: Option<RopeParameters>,
    rope_scaling: Option<RopeParameters>,
    pub vocab_size: usize,
    /// Present on the hybrid Qwen3.5 bases, whose Gated DeltaNet layers this
    /// backbone does not implement.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Qwen3's projections have no bias; a config that says otherwise is not
    /// the architecture implemented here.
    #[serde(default)]
    pub attention_bias: bool,
}

/// The rope block of a config, in either of the two places it appears.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RopeParameters {
    rope_type: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
}

impl RopeParameters {
    pub fn kind(&self) -> &str {
        self.rope_type
            .as_deref()
            .or(self.kind.as_deref())
            .unwrap_or("default")
    }
}

/// Read a `config.json` into whichever config type is asked for.
pub(crate) fn read_config<T: serde::de::DeserializeOwned>(dir: &Path) -> Result<T> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::Engine(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| Error::Engine(format!("cannot parse {}: {e}", path.display())))
}

/// Refuse a rope this backbone would compute differently from the reference.
pub(crate) fn supported_rope(rope: Option<&RopeParameters>) -> Result<()> {
    if let Some(rope) = rope {
        if rope.kind() != "default" {
            return Err(Error::Engine(format!(
                "this backbone implements the original rotary embedding, the config asks for {:?}",
                rope.kind()
            )));
        }
    }
    Ok(())
}

impl Config {
    /// Read `config.json` from a model directory.
    pub fn read(dir: &Path) -> Result<Self> {
        let config: Self = read_config(dir)?;
        config.supported()?;
        Ok(config)
    }

    /// The rotary base. Hugging Face reads it from `rope_parameters` now and
    /// from the top level before that.
    pub fn rope_theta(&self) -> Result<f64> {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|r| r.rope_theta))
            .ok_or_else(|| Error::Engine(String::from("the config states no rope_theta")))
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Refuse a config this backbone would answer wrongly rather than not at
    /// all.
    fn supported(&self) -> Result<()> {
        // Recurrent layers ignore attention masks, so they cannot honour the
        // block-causal mask and need the row form and their own implementation.
        if let Some(types) = &self.layer_types {
            if types.iter().any(|kind| kind != "full_attention") {
                return Err(Error::Engine(String::from(
                    "this backbone is attention-only; the hybrid Qwen3.5 bases \
                     (Gated DeltaNet layers) need the qwen3_5 backbone",
                )));
            }
        }
        if self.use_sliding_window && self.sliding_window.is_some() {
            return Err(Error::Engine(String::from(
                "sliding-window attention is not implemented",
            )));
        }
        // Anything but the original RoPE changes the frequencies, and this
        // backbone computes them one way only.
        supported_rope(self.rope_parameters.as_ref())?;
        supported_rope(self.rope_scaling.as_ref())?;
        if self.attention_bias {
            return Err(Error::Engine(String::from(
                "this backbone reads no attention bias, and the config asks for one",
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(Error::Engine(String::from(
                "the attention heads do not divide into the key/value heads",
            )));
        }
        Ok(())
    }
}

/// A prefilled state: what every question of a request continues from.
///
/// Holds one layer's worth of keys and values per layer, for the state tokens
/// only. Cheap to share — the tensors are read, never written.
pub struct Prefix {
    tokens: Vec<u32>,
    layers: Vec<(Tensor, Tensor)>,
}

impl Prefix {
    /// The state tokens this was prefilled from; the cache key.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// A Qwen3 backbone with the checkpoint's LoRA adapter already merged.
pub struct Backbone {
    config: Config,
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    device: Device,
    dtype: DType,
}

struct Layer {
    input_norm: Tensor,
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_norm: Tensor,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Backbone {
    /// Load a base model, optionally with a LoRA adapter merged into it.
    ///
    /// `base` is a directory of safetensors plus `config.json`; `adapter` is a
    /// PEFT adapter directory (`adapter_config.json`,
    /// `adapter_model.safetensors`), which is what a Kev checkpoint ships.
    pub fn load(
        base: &Path,
        adapter: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let config = Config::read(base)?;
        let weights = Weights::open(base, adapter, device, dtype)?;

        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let layer = format!("layers.{index}");
                Ok(Layer {
                    input_norm: weights.plain(&format!("{layer}.input_layernorm.weight"))?,
                    q_proj: weights.adapted(&format!("{layer}.self_attn.q_proj"))?,
                    k_proj: weights.adapted(&format!("{layer}.self_attn.k_proj"))?,
                    v_proj: weights.adapted(&format!("{layer}.self_attn.v_proj"))?,
                    o_proj: weights.adapted(&format!("{layer}.self_attn.o_proj"))?,
                    // Qwen3 normalises every head's query and key before the
                    // rotary embedding; Qwen2 did not.
                    q_norm: weights.plain(&format!("{layer}.self_attn.q_norm.weight"))?,
                    k_norm: weights.plain(&format!("{layer}.self_attn.k_norm.weight"))?,
                    post_attention_norm: weights
                        .plain(&format!("{layer}.post_attention_layernorm.weight"))?,
                    gate_proj: weights.adapted(&format!("{layer}.mlp.gate_proj"))?,
                    up_proj: weights.adapted(&format!("{layer}.mlp.up_proj"))?,
                    down_proj: weights.adapted(&format!("{layer}.mlp.down_proj"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens: weights.plain("embed_tokens.weight")?,
            norm: weights.plain("norm.weight")?,
            layers,
            config,
            device: device.clone(),
            dtype,
        })
    }

    /// What the backbone runs in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The width of the hidden states this backbone produces.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// One forward pass: the last hidden state of every token.
    ///
    /// `positions` are the position ids, one per token — not `0..n`, since each
    /// question's branch restarts after the state. `mask` is additive and
    /// `[1, 1, n, n]`: `0.0` where a token may read another, very negative
    /// where it may not.
    pub fn forward(&self, ids: &[u32], positions: &[u32], mask: &Tensor) -> Result<Tensor> {
        let (hidden, _) = self.run(&[(ids, positions)], mask, None)?;
        Ok(hidden.i(0)?.to_dtype(DType::F32)?)
    }

    /// Run the state tokens and keep what a branch needs to continue from them.
    ///
    /// The state never attends to a branch (it comes first, and attention is
    /// causal), so its keys and values do not depend on the questions: one
    /// prefill serves every question, and serves the next request with the same
    /// state too. This is the same reuse `kev.serve` does, and exact for the
    /// same reason.
    pub fn prefill(&self, ids: &[u32], positions: &[u32]) -> Result<Prefix> {
        let mask = attention_mask(
            ids.len(),
            |query, key| key <= query,
            &self.device,
            self.dtype,
        )?;
        let (_, layers) = self.run(&[(ids, positions)], &mask, None)?;
        Ok(Prefix {
            tokens: ids.to_vec(),
            layers,
        })
    }

    /// Prefill several states in one pass, padded to the longest.
    ///
    /// Only worth it when the states are short enough that a pass costs more in
    /// overhead than in arithmetic; a long state fills the machine on its own.
    pub fn prefill_batch(&self, rows: &[(&[u32], &[u32])]) -> Result<Vec<Prefix>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, positions, lengths, padded) = pad_rows(rows);
        let mask = prefill_batch_mask(&lengths, padded, &self.device, self.dtype)?;
        let batched: Vec<(&[u32], &[u32])> = (0..rows.len())
            .map(|row| {
                (
                    &ids[row * padded..(row + 1) * padded],
                    &positions[row * padded..(row + 1) * padded],
                )
            })
            .collect();
        let (_, layers) = self.run(&batched, &mask, None)?;

        lengths
            .iter()
            .enumerate()
            .map(|(row, length)| {
                let layers = layers
                    .iter()
                    .map(|(keys, values)| {
                        Ok((
                            keys.narrow(0, row, 1)?
                                .narrow(2, 0, *length)?
                                .contiguous()?,
                            values
                                .narrow(0, row, 1)?
                                .narrow(2, 0, *length)?
                                .contiguous()?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Prefix {
                    tokens: rows[row].0.to_vec(),
                    layers,
                })
            })
            .collect()
    }

    /// The hidden states of one branch, continuing from a prefilled state.
    ///
    /// The branch sees the whole state and its own tokens up to each position,
    /// which is exactly what the packed mask would allow it.
    pub fn forward_from(&self, prefix: &Prefix, ids: &[u32], positions: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_from_batch(prefix, &[(ids, positions)])?
            .remove(0))
    }

    /// The hidden states of several branches at once, all continuing from the
    /// same prefilled state.
    ///
    /// Padded to the longest row and run as one batch: the same arithmetic as one
    /// call each, in a fraction of the calls, which is most of what a short
    /// branch costs on a CPU.
    pub fn forward_from_batch(
        &self,
        prefix: &Prefix,
        rows: &[(&[u32], &[u32])],
    ) -> Result<Vec<Tensor>> {
        let (ids, positions, lengths, padded) = pad_rows(rows);
        let mask = branch_batch_mask(
            prefix.tokens.len(),
            &lengths,
            padded,
            &self.device,
            self.dtype,
        )?;
        let batched: Vec<(&[u32], &[u32])> = (0..rows.len())
            .map(|row| {
                (
                    &ids[row * padded..(row + 1) * padded],
                    &positions[row * padded..(row + 1) * padded],
                )
            })
            .collect();
        let (hidden, _) = self.run(&batched, &mask, Some(prefix))?;
        lengths
            .iter()
            .enumerate()
            .map(|(row, length)| Ok(hidden.i(row)?.narrow(0, 0, *length)?.to_dtype(DType::F32)?))
            .collect()
    }

    /// One pass over a batch of rows: `[rows, tokens, hidden]` out.
    fn run(
        &self,
        rows: &[(&[u32], &[u32])],
        mask: &Tensor,
        prefix: Option<&Prefix>,
    ) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let batch = rows.len();
        let len = rows.first().map(|(ids, _)| ids.len()).unwrap_or(0);
        for (ids, positions) in rows {
            if ids.len() != positions.len() || ids.len() != len {
                return Err(Error::Engine(String::from(
                    "the rows of a batch must be the same length, with one position id each",
                )));
            }
        }
        if let Some(prefix) = prefix {
            if prefix.layers.len() != self.layers.len() {
                return Err(Error::Engine(String::from(
                    "this prefix was prefilled by another model",
                )));
            }
        }

        let ids: Vec<u32> = rows
            .iter()
            .flat_map(|(ids, _)| ids.iter().copied())
            .collect();
        let positions: Vec<u32> = rows
            .iter()
            .flat_map(|(_, positions)| positions.iter().copied())
            .collect();
        let tokens = Tensor::from_slice(&ids, ids.len(), &self.device)?;
        let mut xs = self
            .embed_tokens
            .index_select(&tokens, 0)?
            .reshape((batch, len, self.config.hidden_size))?
            .to_dtype(self.dtype)?;

        let dim = self.config.head_dim();
        let (cos, sin) = rotary_tables(
            &positions,
            dim,
            self.config.rope_theta()?,
            &self.device,
            self.dtype,
        )?;
        let cos = cos.reshape((batch, len, dim / 2))?;
        let sin = sin.reshape((batch, len, dim / 2))?;

        let mut kept = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let past = prefix.map(|prefix| {
                let (k, v) = &prefix.layers[index];
                (k, v)
            });
            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.input_norm, self.config.rms_norm_eps)?;
            let (attended, k, v) = self.attention(layer, &normed, &cos, &sin, mask, past)?;
            kept.push((k, v));
            xs = (residual + attended)?;

            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.post_attention_norm, self.config.rms_norm_eps)?;
            xs = (residual + self.feed_forward(layer, &normed)?)?;
        }

        Ok((rms_norm(&xs, &self.norm, self.config.rms_norm_eps)?, kept))
    }

    /// Attention over a batch of rows, optionally continuing from a state's keys
    /// and values. Returns the output and this pass's keys and values, so a
    /// prefill can keep them.
    fn attention(
        &self,
        layer: &Layer,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, len, _) = xs.dims3()?;
        let heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim();

        let q = linear(xs, &layer.q_proj)?.reshape((batch, len, heads, dim))?;
        let k = linear(xs, &layer.k_proj)?.reshape((batch, len, kv_heads, dim))?;
        let v = linear(xs, &layer.v_proj)?.reshape((batch, len, kv_heads, dim))?;

        // Per-head normalisation, then the rotary embedding, in that order.
        let q = rms_norm(&q, &layer.q_norm, self.config.rms_norm_eps)?;
        let k = rms_norm(&k, &layer.k_norm, self.config.rms_norm_eps)?;
        let q = rope(&q.transpose(1, 2)?, cos, sin)?;
        let k = rope(&k.transpose(1, 2)?, cos, sin)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // The state's keys and values sit in front of this pass's, which is what
        // makes a branch see the whole state and nothing of another branch.
        let (keys, values) = match past {
            None => (k.clone(), v.clone()),
            Some((past_k, past_v)) => {
                let widen = |t: &Tensor| -> Result<Tensor> {
                    let (_, heads, state, dim) = t.dims4()?;
                    Ok(t.expand((batch, heads, state, dim))?.contiguous()?)
                };
                (
                    Tensor::cat(&[widen(past_k)?, k.clone()], 2)?.contiguous()?,
                    Tensor::cat(&[widen(past_v)?, v.clone()], 2)?.contiguous()?,
                )
            }
        };

        let repeats = heads / kv_heads;
        let scale = 1.0 / (dim as f64).sqrt();
        let scores = (q.matmul(&repeat_kv(&keys, repeats)?.transpose(2, 3)?)? * scale)?;
        // The mask is what keeps one question from reading another.
        let scores = scores.broadcast_add(mask)?;
        // The reference takes the softmax in f32 whatever the backbone runs in.
        let weights = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(self.dtype)?;

        let out = weights
            .matmul(&repeat_kv(&values, repeats)?)?
            .transpose(1, 2)?
            .reshape((batch, len, heads * dim))?;
        Ok((linear(&out, &layer.o_proj)?, k, v))
    }

    fn feed_forward(&self, layer: &Layer, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&linear(xs, &layer.gate_proj)?)?;
        let up = linear(xs, &layer.up_proj)?;
        Ok(linear(&(gate * up)?, &layer.down_proj)?)
    }
}
