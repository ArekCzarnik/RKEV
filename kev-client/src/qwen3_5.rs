//! The Qwen3.5 backbone, in candle: attention layers mixed with Gated DeltaNet.
//!
//! This is the generation the current Kev checkpoints are built on
//! (`jaredpalmer/kev-4b` and friends). Three quarters of its layers are not
//! attention at all but a **linear recurrence** — a gated delta rule — and the
//! rest is attention with an output gate and only a quarter of each head
//! rotated. So it is its own backbone, not a variant of [`crate::qwen3`].
//!
//! The recurrence is why Kev runs one question per row on these bases: a
//! recurrent layer carries state forward token by token and cannot be told to
//! skip another question's tokens the way an attention mask can. The rows are
//! independent, so isolation is exact instead of masked — see
//! [`Pass::rows`](crate::Pass::rows).
//!
//! Transcribed from `transformers/models/qwen3_5/modeling_qwen3_5.py`
//! (`Qwen3_5GatedDeltaNet`, `torch_recurrent_gated_delta_rule`,
//! `Qwen3_5Attention`, `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`), in f32, with no
//! cache: one prefill pass per row.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use serde::Deserialize;

use crate::error::{Error, Result};
use crate::qwen3::{read_config, supported_rope, RopeParameters};
use crate::weights::{
    attention_mask, branch_batch_mask, linear, pad_rows, prefill_batch_mask, real_mask, repeat_kv,
    rms_norm, rope, rotary_tables, Weights,
};

/// Hugging Face's default when a Qwen3.5 config does not state one.
const DEFAULT_PARTIAL_ROTARY_FACTOR: f64 = 0.25;

/// The parts of a Qwen3.5 `config.json` this backbone needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    /// `full_attention` or `linear_attention`, one per layer. Without it there
    /// is nothing to tell the two kinds of layer apart.
    pub layer_types: Vec<String>,
    /// The Gated DeltaNet shapes, with Hugging Face's defaults.
    #[serde(default = "four")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "one_twenty_eight")]
    pub linear_key_head_dim: usize,
    #[serde(default = "one_twenty_eight")]
    pub linear_value_head_dim: usize,
    #[serde(default = "sixteen")]
    pub linear_num_key_heads: usize,
    #[serde(default = "thirty_two")]
    pub linear_num_value_heads: usize,
    rope_theta: Option<f64>,
    rope_parameters: Option<RopeParameters>,
    rope_scaling: Option<RopeParameters>,
    #[serde(default)]
    pub attention_bias: bool,
}

fn four() -> usize {
    4
}
fn sixteen() -> usize {
    16
}
fn thirty_two() -> usize {
    32
}
fn one_twenty_eight() -> usize {
    128
}

impl Config {
    pub fn read(dir: &Path) -> Result<Self> {
        let config: Self = read_config(dir)?;
        config.supported()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> Result<f64> {
        self.rope_theta
            .or_else(|| self.rope_parameters.as_ref().and_then(|r| r.rope_theta))
            .ok_or_else(|| Error::Engine(String::from("the config states no rope_theta")))
    }

    /// How much of each head the rotary embedding covers: a quarter by default
    /// on these bases, and the rest of the head passes through unrotated.
    pub fn rotary_dim(&self) -> usize {
        let factor = self
            .rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .unwrap_or(DEFAULT_PARTIAL_ROTARY_FACTOR);
        let dim = (self.head_dim() as f64 * factor) as usize;
        // The tables are built in pairs, and the reference slices at their width.
        dim - dim % 2
    }

    /// `true` when any layer is recurrent, which is what forces the row form.
    pub fn is_hybrid(&self) -> bool {
        self.layer_types
            .iter()
            .any(|kind| kind == "linear_attention")
    }

    fn supported(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Error::Engine(format!(
                "{} layer types for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if let Some(kind) = self
            .layer_types
            .iter()
            .find(|kind| *kind != "full_attention" && *kind != "linear_attention")
        {
            return Err(Error::Engine(format!("unknown layer type {kind:?}")));
        }
        supported_rope(self.rope_parameters.as_ref())?;
        supported_rope(self.rope_scaling.as_ref())?;
        if self.attention_bias {
            return Err(Error::Engine(String::from(
                "this backbone reads no attention bias, and the config asks for one",
            )));
        }
        if self.linear_conv_kernel_dim < 2 {
            return Err(Error::Engine(String::from(
                "a convolution kernel below 2 is not implemented",
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0
            || self.linear_num_value_heads % self.linear_num_key_heads != 0
        {
            return Err(Error::Engine(String::from(
                "the heads do not divide into the key/value heads",
            )));
        }
        Ok(())
    }
}

/// A prefilled state: what every question of a request continues from.
///
/// An attention layer contributes the state's keys and values; a recurrent layer
/// contributes its state matrix and the tail of its convolution window. Neither
/// depends on the questions — the state comes first, and both layer kinds only
/// ever look backwards — so one prefill serves every question of a request, and
/// the next request with the same state. That is the reuse `kev.serve` does, and
/// on these bases it is the difference between running the state once and running
/// it once per question.
pub struct Prefix {
    tokens: Vec<u32>,
    layers: Vec<LayerPrefix>,
}

impl Prefix {
    /// The state tokens this was prefilled from; the cache key.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

enum LayerPrefix {
    Attention {
        keys: Tensor,
        values: Tensor,
    },
    Recurrence {
        /// The last `kernel - 1` columns of the convolution's input.
        window: Tensor,
        /// One state matrix per value head.
        state: Tensor,
    },
}

/// A Qwen3.5 backbone with the checkpoint's LoRA adapter already merged.
pub struct Backbone {
    config: Config,
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    device: Device,
    /// Whether the delta rule runs in chunks; `None` decides per request. See
    /// [`Backbone::with_chunked_recurrence`].
    chunked: Option<bool>,
    /// How many tokens a chunk covers. See [`Backbone::with_chunk_size`].
    chunk: usize,
}

struct Layer {
    input_norm: Tensor,
    post_attention_norm: Tensor,
    mixer: Mixer,
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

enum Mixer {
    Attention(Attention),
    Recurrence(DeltaNet),
}

struct Attention {
    /// Twice as wide as Qwen3's: the second half of every head is an output
    /// gate, not a query.
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

struct DeltaNet {
    /// Depthwise, `[key*2 + value, 1, kernel]`.
    conv1d: Tensor,
    /// One per value head.
    dt_bias: Tensor,
    a_log: Tensor,
    /// The gated output norm, over one value head.
    norm: Tensor,
    in_proj_qkv: Tensor,
    in_proj_z: Tensor,
    in_proj_b: Tensor,
    in_proj_a: Tensor,
    out_proj: Tensor,
}

impl Backbone {
    /// Load a base model, optionally with a Kev adapter merged into it.
    pub fn load(base: &Path, adapter: Option<&Path>, device: &Device) -> Result<Self> {
        let config = Config::read(base)?;
        let weights = Weights::open(base, adapter, device)?;

        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let layer = format!("layers.{index}");
                let mixer = if config.layer_types[index] == "linear_attention" {
                    Mixer::Recurrence(DeltaNet {
                        conv1d: weights.plain(&format!("{layer}.linear_attn.conv1d.weight"))?,
                        dt_bias: weights.plain(&format!("{layer}.linear_attn.dt_bias"))?,
                        a_log: weights.plain(&format!("{layer}.linear_attn.A_log"))?,
                        // Ones-centred, unlike every other norm in this model.
                        norm: weights.plain(&format!("{layer}.linear_attn.norm.weight"))?,
                        in_proj_qkv: weights
                            .adapted(&format!("{layer}.linear_attn.in_proj_qkv"))?,
                        in_proj_z: weights.adapted(&format!("{layer}.linear_attn.in_proj_z"))?,
                        in_proj_b: weights.adapted(&format!("{layer}.linear_attn.in_proj_b"))?,
                        in_proj_a: weights.adapted(&format!("{layer}.linear_attn.in_proj_a"))?,
                        out_proj: weights.adapted(&format!("{layer}.linear_attn.out_proj"))?,
                    })
                } else {
                    Mixer::Attention(Attention {
                        q_proj: weights.adapted(&format!("{layer}.self_attn.q_proj"))?,
                        k_proj: weights.adapted(&format!("{layer}.self_attn.k_proj"))?,
                        v_proj: weights.adapted(&format!("{layer}.self_attn.v_proj"))?,
                        o_proj: weights.adapted(&format!("{layer}.self_attn.o_proj"))?,
                        q_norm: weights.zero_centred_norm(&format!("{layer}.self_attn.q_norm"))?,
                        k_norm: weights.zero_centred_norm(&format!("{layer}.self_attn.k_norm"))?,
                    })
                };
                Ok(Layer {
                    input_norm: weights.zero_centred_norm(&format!("{layer}.input_layernorm"))?,
                    post_attention_norm: weights
                        .zero_centred_norm(&format!("{layer}.post_attention_layernorm"))?,
                    mixer,
                    gate_proj: weights.adapted(&format!("{layer}.mlp.gate_proj"))?,
                    up_proj: weights.adapted(&format!("{layer}.mlp.up_proj"))?,
                    down_proj: weights.adapted(&format!("{layer}.mlp.down_proj"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens: weights.plain("embed_tokens.weight")?,
            norm: weights.zero_centred_norm("norm")?,
            layers,
            config,
            device: device.clone(),
            chunked: None,
            chunk: CHUNK,
        })
    }

    /// Force the chunked form of the delta rule on or off.
    ///
    /// Left alone, it is chosen per request: chunks over 64 tokens, and only when
    /// the value heads are wide enough to pay for the per-chunk algebra. On the
    /// released checkpoints (128-wide keys and values) they are, by a wide
    /// margin; on a toy model they are not, and chunking costs more than it
    /// saves. Both forms produce the same numbers — the difference is measured,
    /// not assumed, and `tests/qwen3_5.rs` has the measurement.
    pub fn with_chunked_recurrence(mut self, chunked: bool) -> Self {
        self.chunked = Some(chunked);
        self
    }

    /// How many tokens one chunk of the delta rule covers. [`CHUNK`] by default,
    /// which is what the reference uses.
    ///
    /// Bigger chunks mean fewer steps in the scan and more work inside each one
    /// — the per-chunk algebra grows with the square of the chunk, the triangular
    /// inverse with its cube — so the best size depends on the checkpoint's
    /// widths and the machine. It has to be a power of two, and at least two:
    /// the block-by-block inverse halves it down to one.
    pub fn with_chunk_size(mut self, tokens: usize) -> Result<Self> {
        if tokens < 2 || !tokens.is_power_of_two() {
            return Err(Error::Engine(format!(
                "a chunk has to be a power of two and at least two tokens, not {tokens}"
            )));
        }
        self.chunk = tokens;
        Ok(self)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// One forward pass over one row: the last hidden state of every token.
    ///
    /// `mask` is the additive mask for the attention layers. The recurrent
    /// layers ignore it — they cannot honour it — so a row must contain exactly
    /// the tokens its question may read, in order, which is what
    /// [`Pass::rows`](crate::Pass::rows) produces.
    pub fn forward(&self, ids: &[u32], positions: &[u32], mask: &Tensor) -> Result<Tensor> {
        let (hidden, _) = self.run(&[(ids, positions)], mask, None, None)?;
        Ok(hidden.i(0)?)
    }

    /// Run the state tokens and keep what a branch needs to continue from them.
    pub fn prefill(&self, ids: &[u32], positions: &[u32]) -> Result<Prefix> {
        let mask = attention_mask(ids.len(), |query, key| key <= query, &self.device)?;
        let (_, layers) = self.run(&[(ids, positions)], &mask, None, None)?;
        Ok(Prefix {
            tokens: ids.to_vec(),
            layers,
        })
    }

    /// Prefill several states in one pass.
    ///
    /// Rows are padded to the longest. What a recurrence does with padding is not
    /// a matter of masking — it walks the tokens — so the decay and the write
    /// strength are zeroed there, which makes a short row hand on exactly the
    /// state it had at its last real token, and the convolution window is taken
    /// at each row's own end.
    pub fn prefill_batch(&self, rows: &[(&[u32], &[u32])]) -> Result<Vec<Prefix>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, positions, lengths, padded) = pad_rows(rows);
        let mask = prefill_batch_mask(&lengths, padded, &self.device)?;
        let batched: Vec<(&[u32], &[u32])> = (0..rows.len())
            .map(|row| {
                (
                    &ids[row * padded..(row + 1) * padded],
                    &positions[row * padded..(row + 1) * padded],
                )
            })
            .collect();
        let (_, layers) = self.run(&batched, &mask, None, Some(&lengths))?;

        let heads = self.config.linear_num_value_heads;
        lengths
            .iter()
            .enumerate()
            .map(|(row, length)| {
                let layers = layers
                    .iter()
                    .map(|layer| {
                        Ok(match layer {
                            LayerPrefix::Attention { keys, values } => LayerPrefix::Attention {
                                keys: keys
                                    .narrow(0, row, 1)?
                                    .narrow(2, 0, *length)?
                                    .contiguous()?,
                                values: values
                                    .narrow(0, row, 1)?
                                    .narrow(2, 0, *length)?
                                    .contiguous()?,
                            },
                            LayerPrefix::Recurrence { window, state } => LayerPrefix::Recurrence {
                                window: window.narrow(0, row, 1)?.contiguous()?,
                                state: state.narrow(0, row * heads, heads)?.contiguous()?,
                            },
                        })
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
    pub fn forward_from(&self, prefix: &Prefix, ids: &[u32], positions: &[u32]) -> Result<Tensor> {
        Ok(self
            .forward_from_batch(prefix, &[(ids, positions)])?
            .remove(0))
    }

    /// The hidden states of several branches at once, all continuing from the
    /// same prefilled state.
    ///
    /// The rows are padded to the longest and run as one batch: the same
    /// arithmetic as one call each, in a fifth of the calls when there are five
    /// questions, which on a CPU is most of what a short branch costs. Pads sit
    /// after every real token, are closed to every real query by the mask, and
    /// are never read back.
    pub fn forward_from_batch(
        &self,
        prefix: &Prefix,
        rows: &[(&[u32], &[u32])],
    ) -> Result<Vec<Tensor>> {
        let (ids, positions, lengths, padded) = pad_rows(rows);
        let state = prefix.tokens.len();
        let mask = branch_batch_mask(state, &lengths, padded, &self.device)?;
        let batched: Vec<(&[u32], &[u32])> = (0..rows.len())
            .map(|row| {
                (
                    &ids[row * padded..(row + 1) * padded],
                    &positions[row * padded..(row + 1) * padded],
                )
            })
            .collect();
        let (hidden, _) = self.run(&batched, &mask, Some(prefix), None)?;
        lengths
            .iter()
            .enumerate()
            .map(|(row, length)| Ok(hidden.i(row)?.narrow(0, 0, *length)?))
            .collect()
    }

    /// One pass over a batch of rows: `[rows, tokens, hidden]` out.
    fn run(
        &self,
        rows: &[(&[u32], &[u32])],
        mask: &Tensor,
        prefix: Option<&Prefix>,
        lengths: Option<&[usize]>,
    ) -> Result<(Tensor, Vec<LayerPrefix>)> {
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
            .to_dtype(DType::F32)?;

        // Every row carries its own position ids, so the tables are built per
        // row and the rotary embedding broadcasts over the heads only.
        let (cos, sin) = rotary_tables(
            &positions,
            self.config.rotary_dim(),
            self.config.rope_theta()?,
            &self.device,
        )?;
        let rotary = self.config.rotary_dim() / 2;
        let cos = cos.reshape((batch, len, rotary))?;
        let sin = sin.reshape((batch, len, rotary))?;

        let eps = self.config.rms_norm_eps;
        let mut kept = Vec::with_capacity(self.layers.len());
        for (index, layer) in self.layers.iter().enumerate() {
            let past = prefix.map(|prefix| &prefix.layers[index]);
            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.input_norm, eps)?;
            let mixed = match (&layer.mixer, past) {
                (Mixer::Attention(attention), past) => {
                    let past = match past {
                        Some(LayerPrefix::Attention { keys, values }) => Some((keys, values)),
                        Some(LayerPrefix::Recurrence { .. }) => {
                            return Err(Error::Engine(String::from(
                                "this prefix has the layers the wrong way round",
                            )))
                        }
                        None => None,
                    };
                    let (mixed, keys, values) =
                        self.attention(attention, &normed, &cos, &sin, mask, past)?;
                    kept.push(LayerPrefix::Attention { keys, values });
                    mixed
                }
                (Mixer::Recurrence(recurrence), past) => {
                    let past = match past {
                        Some(LayerPrefix::Recurrence { window, state }) => Some((window, state)),
                        Some(LayerPrefix::Attention { .. }) => {
                            return Err(Error::Engine(String::from(
                                "this prefix has the layers the wrong way round",
                            )))
                        }
                        None => None,
                    };
                    let (mixed, window, state) =
                        self.recurrence(recurrence, &normed, past, lengths)?;
                    kept.push(LayerPrefix::Recurrence { window, state });
                    mixed
                }
            };
            xs = (residual + mixed)?;

            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.post_attention_norm, eps)?;
            xs = (residual + self.feed_forward(layer, &normed)?)?;
        }

        Ok((rms_norm(&xs, &self.norm, eps)?, kept))
    }

    /// Attention as Qwen3 does it, plus the output gate that comes out of the
    /// second half of `q_proj`, over a batch of rows and optionally continuing
    /// from a state's keys and values.
    fn attention(
        &self,
        layer: &Attention,
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
        let eps = self.config.rms_norm_eps;

        let projected = linear(xs, &layer.q_proj)?.reshape((batch, len, heads, 2 * dim))?;
        let q = projected.narrow(3, 0, dim)?;
        let gate = projected
            .narrow(3, dim, dim)?
            .reshape((batch, len, heads * dim))?;

        let k = linear(xs, &layer.k_proj)?.reshape((batch, len, kv_heads, dim))?;
        let v = linear(xs, &layer.v_proj)?.reshape((batch, len, kv_heads, dim))?;

        let q = rms_norm(&q, &layer.q_norm, eps)?;
        let k = rms_norm(&k, &layer.k_norm, eps)?;
        let q = rope(&q.transpose(1, 2)?, cos, sin)?;
        let k = rope(&k.transpose(1, 2)?, cos, sin)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // The state was prefilled once; every row of this batch reads the same
        // keys and values in front of its own.
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
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        let out = weights
            .matmul(&repeat_kv(&values, repeats)?)?
            .transpose(1, 2)?
            .reshape((batch, len, heads * dim))?;
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        Ok((linear(&out, &layer.o_proj)?, k, v))
    }

    fn recurrence(
        &self,
        layer: &DeltaNet,
        xs: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
        lengths: Option<&[usize]>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, len, _) = xs.dims3()?;
        let config = &self.config;
        let (key_heads, value_heads) = (config.linear_num_key_heads, config.linear_num_value_heads);
        let (key_dim, value_dim) = (config.linear_key_head_dim, config.linear_value_head_dim);
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let channels = 2 * keys + values;
        let kernel = config.linear_conv_kernel_dim;
        let rows = batch * value_heads;

        // Queries, keys and values share one projection and one depthwise
        // convolution over time, which is what makes this a *short* convolution
        // in front of the recurrence rather than an attention.
        let projected = linear(xs, &layer.in_proj_qkv)?
            .transpose(1, 2)?
            .contiguous()?;
        // The convolution reaches `kernel - 1` tokens back. At the start of a
        // sequence that is zeros, which is the same as the reference's padding;
        // continuing from a state, it is the tail the prefix kept.
        let window = match past {
            Some((window, _)) => window.expand((batch, channels, kernel - 1))?.contiguous()?,
            None => Tensor::zeros((batch, channels, kernel - 1), DType::F32, &self.device)?,
        };
        let inputs = Tensor::cat(&[&window, &projected], 2)?.contiguous()?;
        let mixed = inputs.conv1d(&layer.conv1d, 0, 1, 1, channels)?;
        let mixed = candle_nn::ops::silu(&mixed)?.transpose(1, 2)?;
        // What the next branch, or the next request, reaches back into — at each
        // row's own end, not at the padded one.
        let kept_window = match lengths {
            None => inputs.narrow(2, len, kernel - 1)?.contiguous()?,
            Some(lengths) => {
                let windows = lengths
                    .iter()
                    .enumerate()
                    .map(|(row, length)| {
                        Ok(inputs.narrow(0, row, 1)?.narrow(2, *length, kernel - 1)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::cat(&windows, 0)?.contiguous()?
            }
        };

        let q = mixed
            .narrow(2, 0, keys)?
            .reshape((batch, len, key_heads, key_dim))?;
        let k = mixed
            .narrow(2, keys, keys)?
            .reshape((batch, len, key_heads, key_dim))?;
        let v = mixed
            .narrow(2, 2 * keys, values)?
            .reshape((batch, len, value_heads, value_dim))?;
        let z = linear(xs, &layer.in_proj_z)?.reshape((batch, len, value_heads, value_dim))?;

        // beta: how strongly this token overwrites the memory. g: how much of
        // the memory survives it, per head.
        let beta = candle_nn::ops::sigmoid(&linear(xs, &layer.in_proj_b)?)?;
        let a = linear(xs, &layer.in_proj_a)?;
        let decay = (softplus(&a.broadcast_add(&layer.dt_bias)?)?
            .broadcast_mul(&layer.a_log.exp()?)?
            * -1.0)?;

        // Padding is only harmless to a recurrence if it neither decays the
        // state nor writes to it.
        let (beta, decay) = match lengths {
            None => (beta, decay),
            Some(lengths) => {
                let real = real_mask(lengths, len, &self.device)?.unsqueeze(2)?;
                (beta.broadcast_mul(&real)?, decay.broadcast_mul(&real)?)
            }
        };

        // The key heads are shared: each one serves several value heads.
        let group = value_heads / key_heads;
        let q = l2_norm(&repeat_interleave(&q, group)?)?;
        let k = l2_norm(&repeat_interleave(&k, group)?)?;
        // The reference scales the query by the key width, not the value width.
        let q = (q / (key_dim as f64).sqrt())?;

        // Rows and heads both index independent states, so they become one
        // dimension: [rows * heads, tokens, width].
        let heads_first = |x: &Tensor, width: usize| -> candle_core::Result<Tensor> {
            x.transpose(1, 2)?.contiguous()?.reshape((rows, len, width))
        };
        let flatten = |x: &Tensor| -> candle_core::Result<Tensor> {
            x.transpose(1, 2)?.contiguous()?.reshape((rows, len))
        };
        let q = heads_first(&q, key_dim)?;
        let k = heads_first(&k, key_dim)?;
        let v = heads_first(&v, value_dim)?;
        let beta = flatten(&beta)?;
        let decay = flatten(&decay)?;

        let start = match past {
            // Every row starts from the same prefilled state.
            Some((_, state)) => state
                .unsqueeze(0)?
                .expand((batch, value_heads, key_dim, value_dim))?
                .reshape((rows, key_dim, value_dim))?
                .contiguous()?,
            None => Tensor::zeros((rows, key_dim, value_dim), DType::F32, &self.device)?,
        };
        let (out, state) = if self
            .chunked
            .unwrap_or_else(|| chunking_pays(len, key_dim, value_dim))
        {
            delta_rule_chunked(&q, &k, &v, &decay, &beta, start, CHUNK)?
        } else {
            delta_rule_sequential(&q, &k, &v, &decay, &beta, start)?
        };
        // [rows * heads, tokens, value dim] -> [rows, tokens, heads, value dim]
        let out = out
            .reshape((batch, value_heads, len, value_dim))?
            .transpose(1, 2)?;

        // The output norm is gated by z, and normalises one head at a time.
        let out = rms_norm(&out, &layer.norm, config.rms_norm_eps)?;
        let out = (out * candle_nn::ops::silu(&z)?)?.reshape((batch, len, values))?;
        Ok((linear(&out, &layer.out_proj)?, kept_window, state))
    }

    fn feed_forward(&self, layer: &Layer, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&linear(xs, &layer.gate_proj)?)?;
        let up = linear(xs, &layer.up_proj)?;
        Ok(linear(&(gate * up)?, &layer.down_proj)?)
    }
}

/// How many tokens a chunk of the delta rule covers by default. 64 is what the
/// reference uses; [`Backbone::with_chunk_size`] changes it.
pub const CHUNK: usize = 64;

/// Whether to run the delta rule in chunks for this shape.
///
/// Two conditions. There has to be more than a chunk of tokens, or there is
/// nothing to condense. And a chunk's own algebra — a triangular inverse and a
/// few `chunk x chunk` matmuls, so about `chunk^3 / 2` — has to cost less than
/// the `3 * chunk * key * value` the token-by-token form spends over the same
/// span. That puts the crossover at `key * value ~ chunk^2 / 6`: the released
/// checkpoints (128 by 128) are far above it, a toy model far below.
fn chunking_pays(len: usize, key_dim: usize, value_dim: usize) -> bool {
    len > CHUNK && key_dim * value_dim > CHUNK * CHUNK / 6
}

/// The gated delta rule, one token at a time.
///
/// `torch_recurrent_gated_delta_rule`: one state per value head, `[key, value]`.
///
/// ```text
/// S <- S * exp(g_t)                    decay
/// delta <- (v_t - k_t S) * beta_t      how much the memory is off by
/// S <- S + k_t^T delta                 write it back
/// out_t <- q_t S
/// ```
///
/// Exact and obvious, and the form everything else here is checked against.
fn delta_rule_sequential(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    mut state: Tensor,
) -> Result<(Tensor, Tensor)> {
    let (heads, len, _) = q.dims3()?;
    let mut out = Vec::with_capacity(len);
    for step in 0..len {
        let q_t = q.narrow(1, step, 1)?;
        let k_t = k.narrow(1, step, 1)?;
        let v_t = v.narrow(1, step, 1)?;
        let decay_t = decay.narrow(1, step, 1)?.reshape((heads, 1, 1))?.exp()?;
        let beta_t = beta.narrow(1, step, 1)?.reshape((heads, 1, 1))?;

        state = state.broadcast_mul(&decay_t)?;
        let remembered = k_t.matmul(&state)?;
        let delta = (v_t - remembered)?.broadcast_mul(&beta_t)?;
        state = (state + k_t.transpose(1, 2)?.matmul(&delta)?)?;
        out.push(q_t.matmul(&state)?);
    }
    Ok((Tensor::cat(&out, 1)?, state))
}

/// The same rule, a chunk of tokens at a time.
///
/// `torch_chunk_gated_delta_rule`, which is what the reference runs for prefill.
/// Within a chunk the updates are condensed into matmuls through the UT
/// transform — the inverse of a unit lower triangular system — so the sequential
/// scan is left with one step per chunk instead of one per token. Same numbers,
/// and on a long state the difference is the difference between usable and not.
///
/// The inverse is built as `I + A + A^2 + ...` for `A = -strict_lower(system)`,
/// which terminates because `A` is nilpotent, and by doubling — so
/// `log2(chunk)` matmuls rather than `chunk` substitutions.
fn delta_rule_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    mut state: Tensor,
    chunk: usize,
) -> Result<(Tensor, Tensor)> {
    let device = q.device().clone();
    let (heads, len, key_dim) = q.dims3()?;
    let value_dim = v.dim(2)?;
    // The tail is padded with zeros, where a decay of 0 leaves the state alone
    // (exp(0) = 1) and a beta of 0 writes nothing, so the padding cannot change
    // either the outputs or the state it hands on.
    let pad = (chunk - len % chunk) % chunk;
    let chunks = (len + pad) / chunk;
    let pad_time = |x: &Tensor| x.pad_with_zeros(1, 0, pad);

    let q = pad_time(q)?.reshape((heads, chunks, chunk, key_dim))?;
    let k = pad_time(k)?.reshape((heads, chunks, chunk, key_dim))?;
    let v = pad_time(v)?;
    let beta = pad_time(beta)?;
    let decay = pad_time(decay)?.reshape((heads, chunks, chunk))?;

    // beta is how much of the new value is written: apply it to k and v once.
    let v_beta = v
        .broadcast_mul(&beta.unsqueeze(2)?)?
        .reshape((heads, chunks, chunk, value_dim))?;
    let k_beta = k.broadcast_mul(&beta.reshape((heads, chunks, chunk))?.unsqueeze(3)?)?;

    // Decay accumulated from the start of a chunk, and between any two positions
    // in it. In the lower triangle the difference is a sum of decays and so at
    // most zero; the upper triangle is masked away, and clamping keeps its
    // exponential from overflowing on the way.
    let cumulative = decay.cumsum(2)?;
    let pairwise = cumulative
        .unsqueeze(3)?
        .broadcast_sub(&cumulative.unsqueeze(2)?)?
        .clamp(f32::NEG_INFINITY, 0.0)?
        .exp()?;
    let causal = Tensor::tril2(chunk, DType::F32, &device)?.reshape((1, 1, chunk, chunk))?;
    let pairwise = pairwise.broadcast_mul(&causal)?;

    let system = k_beta.matmul(&k.transpose(2, 3)?)?.mul(&pairwise)?;
    let intra = q.matmul(&k.transpose(2, 3)?)?.mul(&pairwise)?;
    let decayed_k_beta = k_beta.broadcast_mul(&cumulative.exp()?.unsqueeze(3)?)?;

    // The UT transform: solve the unit lower triangular system for the values
    // and for what the old state already predicts.
    let strict = Tensor::tril2(chunk, DType::F32, &device)?
        .sub(&Tensor::eye(chunk, DType::F32, &device)?)?
        .reshape((1, 1, chunk, chunk))?;
    let identity = Tensor::eye(chunk, DType::F32, &device)?.reshape((1, 1, chunk, chunk))?;
    let lower = system.broadcast_mul(&strict)?.broadcast_add(&identity)?;
    let inverse = unit_lower_inverse(&lower)?.reshape((heads, chunks, chunk, chunk))?;

    let values = inverse.matmul(&v_beta)?;
    let reads = inverse.matmul(&decayed_k_beta)?;
    // Fold the decays into the queries and keys once, rather than per chunk.
    let q = q.broadcast_mul(&cumulative.exp()?.unsqueeze(3)?)?;
    let last = cumulative.narrow(2, chunk - 1, 1)?;
    let k = k.broadcast_mul(&last.broadcast_sub(&cumulative)?.exp()?.unsqueeze(3)?)?;
    let chunk_decay = last.exp()?;

    let mut out = Vec::with_capacity(chunks);
    for index in 0..chunks {
        let values = values.i((.., index))?;
        let reads = reads.i((.., index))?;
        // What this chunk writes, minus what the state already predicted.
        let corrected = (values - reads.matmul(&state)?)?;
        let between = q.i((.., index))?.matmul(&state)?;
        out.push((between + intra.i((.., index))?.matmul(&corrected)?)?);
        state = (state.broadcast_mul(&chunk_decay.i((.., index))?.reshape((heads, 1, 1))?)?
            + k.i((.., index))?.transpose(1, 2)?.matmul(&corrected)?)?;
    }

    let out = Tensor::cat(&out, 1)?.narrow(1, 0, len)?;
    Ok((out, state))
}

/// The inverse of a unit lower triangular matrix, block by block.
///
/// ```text
/// inv([[A, 0], [B, C]]) = [[inv A, 0], [-inv C * B * inv A, inv C]]
/// ```
///
/// The two diagonal blocks are independent, so they go into the batch dimension
/// and one recursive call does both: `log2(n)` levels, a handful of matmuls each.
/// Building the inverse as `I + A + A^2 + ...` instead would be simpler and cost
/// six times the arithmetic — which on the released checkpoints' widths is the
/// difference between chunking being worth it and not.
///
/// `n` has to be a power of two, which [`CHUNK`] is.
fn unit_lower_inverse(l: &Tensor) -> candle_core::Result<Tensor> {
    let dims = l.dims();
    let n = dims[dims.len() - 1];
    let batch: usize = dims[..dims.len() - 2].iter().product();
    let l = l.reshape((batch, n, n))?;
    if n == 1 {
        // A one-by-one unit lower triangular matrix is its own inverse.
        return Tensor::ones((batch, 1, 1), l.dtype(), l.device());
    }

    let half = n / 2;
    let top_left = l.narrow(1, 0, half)?.narrow(2, 0, half)?;
    let bottom_left = l.narrow(1, half, half)?.narrow(2, 0, half)?.contiguous()?;
    let bottom_right = l.narrow(1, half, half)?.narrow(2, half, half)?;

    let blocks = Tensor::cat(&[top_left.unsqueeze(1)?, bottom_right.unsqueeze(1)?], 1)?
        .reshape((batch * 2, half, half))?;
    let inverses = unit_lower_inverse(&blocks.contiguous()?)?.reshape((batch, 2, half, half))?;
    let upper = inverses.i((.., 0))?.contiguous()?;
    let lower = inverses.i((.., 1))?.contiguous()?;
    let corner = lower.matmul(&bottom_left)?.matmul(&upper)?.neg()?;

    let zeros = Tensor::zeros((batch, half, half), l.dtype(), l.device())?;
    Tensor::cat(
        &[
            Tensor::cat(&[upper, zeros], 2)?,
            Tensor::cat(&[corner, lower], 2)?,
        ],
        1,
    )
}

/// `x / sqrt(sum(x^2) + eps)` over the last dimension, the way the FLA library
/// normalises queries and keys (`l2norm`, eps 1e-6) — not an RMS norm.
fn l2_norm(xs: &Tensor) -> candle_core::Result<Tensor> {
    let inverse = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.powf(-0.5)?;
    xs.broadcast_mul(&inverse)
}

/// `ln(1 + exp(x))`, in the form that does not overflow for large `x`.
fn softplus(xs: &Tensor) -> candle_core::Result<Tensor> {
    let stable = ((xs.abs()?.neg()?.exp()? + 1.0)?).log()?;
    xs.relu()? + stable
}

/// Every head repeated `n` times in place, as `repeat_interleave(n, dim=2)`
/// does: head `h` of the input serves heads `h*n .. h*n+n` of the output.
fn repeat_interleave(xs: &Tensor, n: usize) -> candle_core::Result<Tensor> {
    if n == 1 {
        return xs.contiguous();
    }
    let (batch, len, heads, dim) = xs.dims4()?;
    xs.unsqueeze(3)?
        .expand((batch, len, heads, n, dim))?
        .reshape((batch, len, heads * n, dim))
}
