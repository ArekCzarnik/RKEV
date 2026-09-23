//! Reading a checkpoint's tensors, with its LoRA adapter merged in.
//!
//! Shared by the backbones: a Kev checkpoint is always a base model plus a
//! rank-16 adapter, and merging it at load time in f32 is what the Python does
//! (`LoadOptions.merge`), exactly there.

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::error::{Error, Result};

/// The checkpoint on disk: base weights, and the adapter folded into them as
/// they are read.
pub(crate) struct Weights {
    base: candle_core::safetensors::MmapedSafetensors,
    adapter: Option<Adapter>,
    device: Device,
}

struct Adapter {
    tensors: candle_core::safetensors::MmapedSafetensors,
    /// `lora_alpha / r`, the scale peft folds into the delta.
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
    pub fn open(base: &Path, adapter: Option<&Path>, device: &Device) -> Result<Self> {
        let files = safetensors_in(base)?;
        // Safety: the files must not change while they are mapped, which is the
        // same contract every safetensors reader takes.
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

    /// A tensor the adapter never touches (the embeddings, the norms, the
    /// convolution and the per-head scalars).
    pub fn plain(&self, path: &str) -> Result<Tensor> {
        let name = format!("model.{path}");
        Ok(self.base.load(&name, &self.device)?.to_dtype(DType::F32)?)
    }

    /// A norm weight, which is stored as the deviation from 1.0 in the Qwen3.5
    /// bases (`Qwen3_5RMSNorm` computes `x * (1 + weight)` from a zero-centred
    /// parameter). Folding the 1.0 in here keeps the forward pass to one
    /// multiplication, as candle's `rms_norm` expects.
    pub fn zero_centred_norm(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain(&format!("{path}.weight"))?;
        Ok((weight + 1.0)?)
    }

    /// A projection, with the adapter's `B @ A` delta merged in.
    pub fn adapted(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain(&format!("{path}.weight"))?;
        let Some(adapter) = &self.adapter else {
            return Ok(weight);
        };
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
        // peft names the module it wrapped `base_model.model.<path>`; kev wraps
        // the text model, so `<path>` is what the base file calls `model.<path>`.
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

fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
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

/// `y = x W^T`, for the bias-free projections these models use throughout.
pub(crate) fn linear(xs: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
    xs.broadcast_matmul(&weight.t()?)
}

pub(crate) fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    candle_nn::ops::rms_norm(&xs.contiguous()?, weight, eps as f32)
}

/// Grouped-query attention: every key/value head serves `n` query heads.
pub(crate) fn repeat_kv(xs: &Tensor, n: usize) -> candle_core::Result<Tensor> {
    if n == 1 {
        return Ok(xs.clone());
    }
    let (batch, heads, len, dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((batch, heads, n, len, dim))?
        .reshape((batch, heads * n, len, dim))
}

/// Cosine and sine tables for exactly the positions asked for, rather than for
/// `0..n` — the branches do not sit at consecutive positions.
///
/// `inv_freq[i] = theta^(-2i/dim)`, as in Hugging Face's
/// `compute_default_rope_parameters`, where `dim` is the *rotary* width: the
/// whole head on Qwen3, a quarter of it on Qwen3.5. The tables are `[len,
/// dim/2]`, which is what candle's non-interleaved `rope` wants, and the halves
/// it pairs are the same ones `rotate_half` pairs.
pub(crate) fn rotary_tables(
    positions: &[u32],
    dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let inverse: Vec<f32> = (0..dim / 2)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let inverse = Tensor::from_vec(inverse, (1, dim / 2), device)?;
    let angles: Vec<f32> = positions.iter().map(|p| *p as f32).collect();
    let angles = Tensor::from_vec(angles, (positions.len(), 1), device)?;
    let angles = angles.matmul(&inverse)?;
    Ok((angles.cos()?, angles.sin()?))
}

/// Apply the rotary embedding to the first `2 * cos.dim(1)` components of each
/// head, leaving the rest as they are.
///
/// Qwen3 rotates the whole head; Qwen3.5 rotates a quarter of it
/// (`partial_rotary_factor`) and passes the remainder through, as
/// `apply_rotary_pos_emb` does by slicing at `cos.shape[-1]`.
pub(crate) fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let rotary = 2 * cos.dim(1)?;
    let dim = xs.dim(3)?;
    let xs = xs.contiguous()?;
    if rotary == dim {
        return candle_nn::rotary_emb::rope(&xs, cos, sin);
    }
    let rotated = candle_nn::rotary_emb::rope(&xs.narrow(3, 0, rotary)?.contiguous()?, cos, sin)?;
    Tensor::cat(&[rotated, xs.narrow(3, rotary, dim - rotary)?], 3)
}

/// The additive mask for a branch pass continuing from a prefilled state,
/// `[1, 1, branch, state + branch]`.
///
/// The state is visible from every branch position — it came first and cannot
/// contain another question — and the branch is causal within itself.
pub(crate) fn branch_mask(state: usize, branch: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(branch * (state + branch));
    for query in 0..branch {
        for key in 0..state + branch {
            let allowed = key < state || key - state <= query;
            values.push(if allowed { 0.0 } else { f32::MIN });
        }
    }
    Ok(Tensor::from_vec(
        values,
        (1, 1, branch, state + branch),
        device,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rotary embedding, against the formula in
    /// `transformers/models/qwen3/modeling_qwen3.py`:
    ///
    /// ```text
    /// inv_freq[i] = theta ** (-2i/dim)            (compute_default_rope_parameters)
    /// cos = cat(freqs, freqs).cos()               (Qwen3RotaryEmbedding.forward)
    /// rotate_half(x) = cat(-x[d/2:], x[:d/2])
    /// q' = q * cos + rotate_half(q) * sin         (apply_rotary_pos_emb)
    /// ```
    ///
    /// Which comes out as, for `i < dim/2`:
    ///   `q'[i] = q[i] cos - q[i + dim/2] sin`
    ///   `q'[i + dim/2] = q[i + dim/2] cos + q[i] sin`
    ///
    /// The point of testing it: candle offers both this and the *interleaved*
    /// convention (`rope_i`), which pairs `(0,1), (2,3), ...` instead. Both run,
    /// only one is Qwen3, and the difference is invisible without real weights.
    /// The partial case is Qwen3.5's, where the tail of every head is left alone.
    #[test]
    fn the_rotary_embedding_matches_the_hugging_face_formula() {
        let device = Device::Cpu;
        let (heads, len, dim, theta) = (2usize, 3usize, 8usize, 10_000f64);
        let positions = [0u32, 5, 9];
        let values: Vec<f32> = (0..heads * len * dim)
            .map(|i| ((i * 37 % 19) as f32 - 9.0) / 7.0)
            .collect();
        let q = Tensor::from_vec(values.clone(), (1, heads, len, dim), &device).unwrap();

        for rotary in [dim, dim / 2] {
            let (cos, sin) = rotary_tables(&positions, rotary, theta, &device).unwrap();
            let ours = rope(&q, &cos, &sin)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            for head in 0..heads {
                for (step, position) in positions.iter().enumerate() {
                    let row = (head * len + step) * dim;
                    for i in 0..rotary / 2 {
                        let angle = *position as f64 * theta.powf(-2.0 * i as f64 / rotary as f64);
                        let (cos, sin) = (angle.cos() as f32, angle.sin() as f32);
                        let (low, high) = (values[row + i], values[row + i + rotary / 2]);
                        let expected = [low * cos - high * sin, high * cos + low * sin];
                        for (offset, expected) in [(i, expected[0]), (i + rotary / 2, expected[1])]
                        {
                            let got = ours[row + offset];
                            assert!(
                                (got - expected).abs() < 1e-6,
                                "rotary {rotary}, position {position}, element {offset}: \
                                 {got} vs {expected}"
                            );
                        }
                    }
                    // Beyond the rotary width, Qwen3.5 passes the head through.
                    for i in rotary..dim {
                        assert_eq!(ours[row + i], values[row + i], "element {i} was rotated");
                    }
                }
            }
        }
    }
}
