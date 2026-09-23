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
    pub rope_theta: f64,
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

impl Config {
    /// Read `config.json` from a model directory.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Engine(format!("cannot read {}: {e}", path.display())))?;
        let config: Self = serde_json::from_str(&text)
            .map_err(|e| Error::Engine(format!("cannot parse {}: {e}", path.display())))?;
        config.supported()?;
        Ok(config)
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
                     (Gated DeltaNet layers) are not implemented",
                )));
            }
        }
        if self.use_sliding_window && self.sliding_window.is_some() {
            return Err(Error::Engine(String::from(
                "sliding-window attention is not implemented",
            )));
        }
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

/// A Qwen3 backbone with the checkpoint's LoRA adapter already merged in.
pub struct Backbone {
    config: Config,
    embed_tokens: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    device: Device,
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
    pub fn load(base: &Path, adapter: Option<&Path>, device: &Device) -> Result<Self> {
        let config = Config::read(base)?;
        let weights = Weights::open(base, adapter, device)?;

        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let layer = format!("layers.{index}");
                Ok(Layer {
                    input_norm: weights.plain(&format!("{layer}.input_layernorm"))?,
                    q_proj: weights.adapted(&format!("{layer}.self_attn.q_proj"))?,
                    k_proj: weights.adapted(&format!("{layer}.self_attn.k_proj"))?,
                    v_proj: weights.adapted(&format!("{layer}.self_attn.v_proj"))?,
                    o_proj: weights.adapted(&format!("{layer}.self_attn.o_proj"))?,
                    // Qwen3 normalises every head's query and key before the
                    // rotary embedding; Qwen2 did not.
                    q_norm: weights.plain(&format!("{layer}.self_attn.q_norm"))?,
                    k_norm: weights.plain(&format!("{layer}.self_attn.k_norm"))?,
                    post_attention_norm: weights
                        .plain(&format!("{layer}.post_attention_layernorm"))?,
                    gate_proj: weights.adapted(&format!("{layer}.mlp.gate_proj"))?,
                    up_proj: weights.adapted(&format!("{layer}.mlp.up_proj"))?,
                    down_proj: weights.adapted(&format!("{layer}.mlp.down_proj"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embed_tokens: weights.plain("embed_tokens")?,
            norm: weights.plain("norm")?,
            layers,
            config,
            device: device.clone(),
        })
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
        if ids.len() != positions.len() {
            return Err(Error::Engine(format!(
                "{} tokens but {} position ids",
                ids.len(),
                positions.len()
            )));
        }
        let tokens = Tensor::from_slice(ids, ids.len(), &self.device)?;
        let mut xs = self
            .embed_tokens
            .index_select(&tokens, 0)?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?;

        let (cos, sin) = self.rotary(positions)?;
        for layer in &self.layers {
            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.input_norm, self.config.rms_norm_eps)?;
            xs = (residual + self.attention(layer, &normed, &cos, &sin, mask)?)?;

            let residual = xs.clone();
            let normed = rms_norm(&xs, &layer.post_attention_norm, self.config.rms_norm_eps)?;
            xs = (residual + self.feed_forward(layer, &normed)?)?;
        }

        Ok(rms_norm(&xs, &self.norm, self.config.rms_norm_eps)?.i(0)?)
    }

    fn attention(
        &self,
        layer: &Layer,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (_, len, _) = xs.dims3()?;
        let heads = self.config.num_attention_heads;
        let kv_heads = self.config.num_key_value_heads;
        let dim = self.config.head_dim();

        let q = linear(xs, &layer.q_proj)?.reshape((1, len, heads, dim))?;
        let k = linear(xs, &layer.k_proj)?.reshape((1, len, kv_heads, dim))?;
        let v = linear(xs, &layer.v_proj)?.reshape((1, len, kv_heads, dim))?;

        // Per-head normalisation, then the rotary embedding, in that order.
        let q = rms_norm(&q, &layer.q_norm, self.config.rms_norm_eps)?;
        let k = rms_norm(&k, &layer.k_norm, self.config.rms_norm_eps)?;
        let q = candle_nn::rotary_emb::rope(&q.transpose(1, 2)?.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.transpose(1, 2)?.contiguous()?, cos, sin)?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let k = repeat_kv(&k, heads / kv_heads)?;
        let v = repeat_kv(&v, heads / kv_heads)?;

        let scale = 1.0 / (dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        // The mask is what keeps one question from reading another.
        let scores = scores.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;

        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((1, len, heads * dim))?;
        Ok(linear(&out, &layer.o_proj)?)
    }

    fn feed_forward(&self, layer: &Layer, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&linear(xs, &layer.gate_proj)?)?;
        let up = linear(xs, &layer.up_proj)?;
        Ok(linear(&(gate * up)?, &layer.down_proj)?)
    }

    /// Rotary tables for exactly the positions asked for, rather than for
    /// `0..n` — the branches do not sit at consecutive positions.
    fn rotary(&self, positions: &[u32]) -> Result<(Tensor, Tensor)> {
        let dim = self.config.head_dim();
        let inverse: Vec<f32> = (0..dim / 2)
            .map(|i| (1.0 / self.config.rope_theta.powf(2.0 * i as f64 / dim as f64)) as f32)
            .collect();
        let inverse = Tensor::from_vec(inverse, (1, dim / 2), &self.device)?;
        let positions: Vec<f32> = positions.iter().map(|p| *p as f32).collect();
        let positions = Tensor::from_vec(positions.clone(), (positions.len(), 1), &self.device)?;
        let angles = positions.matmul(&inverse)?;
        Ok((angles.cos()?, angles.sin()?))
    }
}

/// `y = x W^T`. Qwen3's projections carry no bias.
fn linear(xs: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
    xs.broadcast_matmul(&weight.t()?)
}

fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    candle_nn::ops::rms_norm(&xs.contiguous()?, weight, eps as f32)
}

/// Grouped-query attention: every key/value head serves `n` query heads.
fn repeat_kv(xs: &Tensor, n: usize) -> candle_core::Result<Tensor> {
    if n == 1 {
        return Ok(xs.clone());
    }
    let (batch, heads, len, dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, heads, n, len, dim))?
        .reshape((batch, heads * n, len, dim))
}

/// The checkpoint on disk: base weights, and the LoRA adapter merged into them
/// as they are read.
struct Weights {
    base: candle_core::safetensors::MmapedSafetensors,
    adapter: Option<Adapter>,
    device: Device,
}

struct Adapter {
    tensors: candle_core::safetensors::MmapedSafetensors,
    /// `lora_alpha / r`, the scale PEFT folds into the delta.
    scale: f64,
}

#[derive(Deserialize)]
struct AdapterConfig {
    r: f64,
    lora_alpha: f64,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    trainable_token_indices: Option<serde_json::Value>,
}

impl Weights {
    fn open(base: &Path, adapter: Option<&Path>, device: &Device) -> Result<Self> {
        let files = safetensors_in(base)?;
        // Safety: the files must not change while they are mapped, which is
        // the same contract every safetensors reader takes.
        let base = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&files)? };

        let adapter = match adapter {
            None => None,
            Some(dir) => {
                let path = dir.join("adapter_config.json");
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| Error::Engine(format!("cannot read {}: {e}", path.display())))?;
                let config: AdapterConfig = serde_json::from_str(&text)
                    .map_err(|e| Error::Engine(format!("cannot parse {}: {e}", path.display())))?;
                if config.trainable_token_indices.is_some() {
                    return Err(Error::Engine(String::from(
                        "this adapter carries trained token embeddings, which are not merged here",
                    )));
                }
                let divisor = if config.use_rslora {
                    config.r.sqrt()
                } else {
                    config.r
                };
                Some(Adapter {
                    tensors: unsafe {
                        candle_core::safetensors::MmapedSafetensors::new(
                            dir.join("adapter_model.safetensors"),
                        )?
                    },
                    scale: config.lora_alpha / divisor,
                })
            }
        };

        Ok(Self {
            base,
            adapter,
            device: device.clone(),
        })
    }

    /// A weight the adapter never touches (the embeddings and the norms).
    fn plain(&self, path: &str) -> Result<Tensor> {
        let name = format!("model.{path}.weight");
        Ok(self.base.load(&name, &self.device)?.to_dtype(DType::F32)?)
    }

    /// A projection, with the adapter's `B @ A` delta merged in.
    ///
    /// Merging in f32 before anything else is what the Python does
    /// (`LoadOptions.merge`), and it is exact there.
    fn adapted(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain(path)?;
        let Some(adapter) = &self.adapter else {
            return Ok(weight);
        };
        // peft names the module it wrapped `base_model.model.<path>`; kev wraps
        // the text model, so `<path>` is what the base file calls `model.<path>`.
        let Some((a, b)) = self.lora(adapter, path)? else {
            return Ok(weight);
        };
        let delta = (b.matmul(&a)? * adapter.scale)?;
        if delta.dims() != weight.dims() {
            return Err(Error::Engine(format!(
                "the adapter's delta for {path} is {:?}, the weight is {:?}",
                delta.dims(),
                weight.dims()
            )));
        }
        Ok((weight + delta)?)
    }

    fn lora(&self, adapter: &Adapter, path: &str) -> Result<Option<(Tensor, Tensor)>> {
        for prefix in ["base_model.model", "base_model.model.model"] {
            let a = format!("{prefix}.{path}.lora_A.weight");
            let b = format!("{prefix}.{path}.lora_B.weight");
            if let Ok(a) = adapter.tensors.load(&a, &self.device) {
                let b = adapter.tensors.load(&b, &self.device)?;
                return Ok(Some((a.to_dtype(DType::F32)?, b.to_dtype(DType::F32)?)));
            }
        }
        Ok(None)
    }
}

fn safetensors_in(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| Error::Engine(format!("cannot read {}: {e}", dir.display())))?;
    let mut files: Vec<_> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    if files.is_empty() {
        return Err(Error::Engine(format!(
            "no .safetensors in {}",
            dir.display()
        )));
    }
    // Shards are named model-00001-of-00002.safetensors; read them in order.
    files.sort();
    Ok(files)
}

/// The additive attention mask for one pass, `[1, 1, n, n]`.
///
/// `allowed(query, key)` is [`Pass::attends`](crate::Pass): causal, and blind
/// across questions.
pub(crate) fn attention_mask<F>(len: usize, allowed: F, device: &Device) -> Result<Tensor>
where
    F: Fn(usize, usize) -> bool,
{
    let mut values = Vec::with_capacity(len * len);
    for query in 0..len {
        for key in 0..len {
            // The same "very negative" the Python uses (torch.finfo.min), not
            // -inf: a fully masked row would otherwise be NaN rather than flat.
            values.push(if allowed(query, key) { 0.0 } else { f32::MIN });
        }
    }
    Ok(Tensor::from_vec(values, (1, 1, len, len), device)?)
}
