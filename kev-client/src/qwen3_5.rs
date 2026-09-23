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
    attention_mask, branch_mask, linear, repeat_kv, rms_norm, rope, rotary_tables, Weights,
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
        })
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
        Ok(self.run(ids, positions, mask, None)?.0)
    }

    /// Run the state tokens and keep what a branch needs to continue from them.
    pub fn prefill(&self, ids: &[u32], positions: &[u32]) -> Result<Prefix> {
        let mask = attention_mask(ids.len(), |query, key| key <= query, &self.device)?;
        let (_, layers) = self.run(ids, positions, &mask, None)?;
        Ok(Prefix {
            tokens: ids.to_vec(),
            layers,
        })
    }

    /// The hidden states of one branch, continuing from a prefilled state.
    pub fn forward_from(&self, prefix: &Prefix, ids: &[u32], positions: &[u32]) -> Result<Tensor> {
        let mask = branch_mask(prefix.tokens.len(), ids.len(), &self.device)?;
        Ok(self.run(ids, positions, &mask, Some(prefix))?.0)
    }

    fn run(
        &self,
        ids: &[u32],
        positions: &[u32],
        mask: &Tensor,
        prefix: Option<&Prefix>,
    ) -> Result<(Tensor, Vec<LayerPrefix>)> {
        if ids.len() != positions.len() {
            return Err(Error::Engine(format!(
                "{} tokens but {} position ids",
                ids.len(),
                positions.len()
            )));
        }
        if let Some(prefix) = prefix {
            if prefix.layers.len() != self.layers.len() {
                return Err(Error::Engine(String::from(
                    "this prefix was prefilled by another model",
                )));
            }
        }
        let tokens = Tensor::from_slice(ids, ids.len(), &self.device)?;
        let mut xs = self
            .embed_tokens
            .index_select(&tokens, 0)?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;

        let (cos, sin) = rotary_tables(
            positions,
            self.config.rotary_dim(),
            self.config.rope_theta()?,
            &self.device,
        )?;
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
                    let (mixed, window, state) = self.recurrence(recurrence, &normed, past)?;
                    kept.push(LayerPrefix::Recurrence { window, state });
                    mixed
                }
            };
            xs = (residual + mixed)?;

            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.post_attention_norm, eps)?;
            xs = (residual + self.feed_forward(layer, &normed)?)?;
        }

        Ok((rms_norm(&xs, &self.norm, eps)?.i(0)?, kept))
    }

    /// Attention as Qwen3 does it, plus the output gate that comes out of the
    /// second half of `q_proj`, and optionally continuing from a state's keys
    /// and values.
    fn attention(
        &self,
        layer: &Attention,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, len, _) = xs.dims3()?;
        let heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim();
        let eps = self.config.rms_norm_eps;

        let projected = linear(xs, &layer.q_proj)?.reshape((1, len, heads, 2 * dim))?;
        let q = projected.narrow(3, 0, dim)?;
        let gate = projected
            .narrow(3, dim, dim)?
            .reshape((1, len, heads * dim))?;

        let k = linear(xs, &layer.k_proj)?.reshape((1, len, kv_heads, dim))?;
        let v = linear(xs, &layer.v_proj)?.reshape((1, len, kv_heads, dim))?;

        let q = rms_norm(&q, &layer.q_norm, eps)?;
        let k = rms_norm(&k, &layer.k_norm, eps)?;
        let q = rope(&q.transpose(1, 2)?, cos, sin)?;
        let k = rope(&k.transpose(1, 2)?, cos, sin)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (keys, values) = match past {
            None => (k.clone(), v.clone()),
            Some((past_k, past_v)) => (
                Tensor::cat(&[past_k, &k], 2)?.contiguous()?,
                Tensor::cat(&[past_v, &v], 2)?.contiguous()?,
            ),
        };

        let repeats = heads / kv_heads;
        let scale = 1.0 / (dim as f64).sqrt();
        let scores = (q.matmul(&repeat_kv(&keys, repeats)?.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        let out = weights
            .matmul(&repeat_kv(&values, repeats)?)?
            .transpose(1, 2)?
            .reshape((1, len, heads * dim))?;
        let out = (out * candle_nn::ops::sigmoid(&gate)?)?;
        Ok((linear(&out, &layer.o_proj)?, k, v))
    }

    /// The gated delta rule, token by token.
    ///
    /// `torch_recurrent_gated_delta_rule`, which the reference uses for single
    /// tokens and whose chunked twin it uses for prefill; they compute the same
    /// thing. One state per value head, `[key dim, value dim]`:
    ///
    /// ```text
    /// S <- S * exp(g_t)                    decay
    /// delta <- (v_t - k_t S) * beta_t      how much the memory is off by
    /// S <- S + k_t^T delta                 write it back
    /// out_t <- q_t S
    /// ```
    fn recurrence(
        &self,
        layer: &DeltaNet,
        xs: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (_, len, _) = xs.dims3()?;
        let config = &self.config;
        let (key_heads, value_heads) = (config.linear_num_key_heads, config.linear_num_value_heads);
        let (key_dim, value_dim) = (config.linear_key_head_dim, config.linear_value_head_dim);
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let channels = 2 * keys + values;
        let kernel = config.linear_conv_kernel_dim;

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
            Some((window, _)) => window.clone(),
            None => Tensor::zeros((1, channels, kernel - 1), DType::F32, &self.device)?,
        };
        let inputs = Tensor::cat(&[&window, &projected], 2)?.contiguous()?;
        let mixed = inputs.conv1d(&layer.conv1d, 0, 1, 1, channels)?;
        let mixed = candle_nn::ops::silu(&mixed)?.transpose(1, 2)?;
        // What the next branch, or the next request, reaches back into.
        let kept_window = inputs.narrow(2, len, kernel - 1)?.contiguous()?;

        let q = mixed
            .narrow(2, 0, keys)?
            .reshape((1, len, key_heads, key_dim))?;
        let k = mixed
            .narrow(2, keys, keys)?
            .reshape((1, len, key_heads, key_dim))?;
        let v = mixed
            .narrow(2, 2 * keys, values)?
            .reshape((1, len, value_heads, value_dim))?;
        let z = linear(xs, &layer.in_proj_z)?.reshape((1, len, value_heads, value_dim))?;

        // beta: how strongly this token overwrites the memory. g: how much of
        // the memory survives it, per head.
        let beta = candle_nn::ops::sigmoid(&linear(xs, &layer.in_proj_b)?)?;
        let a = linear(xs, &layer.in_proj_a)?;
        let decay = (softplus(&a.broadcast_add(&layer.dt_bias)?)?
            .broadcast_mul(&layer.a_log.exp()?)?
            * -1.0)?;

        // The key heads are shared: each one serves several value heads.
        let group = value_heads / key_heads;
        let q = l2_norm(&repeat_interleave(&q, group)?)?;
        let k = l2_norm(&repeat_interleave(&k, group)?)?;
        // The reference scales the query by the key width, not the value width.
        let q = (q / (key_dim as f64).sqrt())?;

        let mut state = match past {
            Some((_, state)) => state.clone(),
            None => Tensor::zeros((value_heads, key_dim, value_dim), DType::F32, &self.device)?,
        };
        let mut out = Vec::with_capacity(len);
        for step in 0..len {
            let at = |tensor: &Tensor, width: usize| -> Result<Tensor> {
                Ok(tensor
                    .narrow(1, step, 1)?
                    .reshape((value_heads, 1, width))?
                    .contiguous()?)
            };
            let q_t = at(&q, key_dim)?;
            let k_t = at(&k, key_dim)?;
            let v_t = at(&v, value_dim)?;
            let decay_t = decay
                .narrow(1, step, 1)?
                .reshape((value_heads, 1, 1))?
                .exp()?;
            let beta_t = beta.narrow(1, step, 1)?.reshape((value_heads, 1, 1))?;

            state = state.broadcast_mul(&decay_t)?;
            let remembered = k_t.matmul(&state)?;
            let delta = (v_t - remembered)?.broadcast_mul(&beta_t)?;
            state = (state + k_t.transpose(1, 2)?.matmul(&delta)?)?;
            out.push(q_t.matmul(&state)?);
        }

        // [value heads, len, value dim] -> [1, len, value heads, value dim]
        let out = Tensor::cat(&out, 1)?.transpose(0, 1)?.unsqueeze(0)?;
        // The output norm is gated by z, and normalises one head at a time.
        let out = rms_norm(&out, &layer.norm, config.rms_norm_eps)?;
        let out = (out * candle_nn::ops::silu(&z)?)?.reshape((1, len, values))?;
        Ok((linear(&out, &layer.out_proj)?, kept_window, state))
    }

    fn feed_forward(&self, layer: &Layer, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&linear(xs, &layer.gate_proj)?)?;
        let up = linear(xs, &layer.up_proj)?;
        Ok(linear(&(gate * up)?, &layer.down_proj)?)
    }
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
