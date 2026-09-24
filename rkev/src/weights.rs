//! Reading a checkpoint's tensors, with its LoRA adapter merged in.
//!
//! Shared by the backbones: a Kev checkpoint is always a base model plus a
//! rank-16 adapter, and merging it at load time in f32 is what the Python does
//! (`LoadOptions.merge`), exactly there.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

use crate::error::{Error, Result};

/// Set to any non-empty value to turn a refusal about tensors this crate does not
/// use into a warning — in the adapter ([`Weights::adapter_fully_merged`]) and in
/// `head.pt` alike. For a checkpoint that carries something this crate does not
/// implement, once you have read the names it printed and decided they cannot
/// reach an answer.
pub(crate) const ALLOW_UNUSED: &str = "KEV_ALLOW_UNUSED";

/// Whether [`ALLOW_UNUSED`] is set to something.
pub(crate) fn unused_tensors_allowed() -> bool {
    std::env::var_os(ALLOW_UNUSED).is_some_and(|value| !value.is_empty())
}

/// The checkpoint on disk: base weights, and the adapter folded into them as
/// they are read.
pub(crate) struct Weights {
    base: candle_core::safetensors::MmapedSafetensors,
    adapter: Option<Adapter>,
    device: Device,
    /// What the backbone runs in. The adapter is merged in f32 whatever this
    /// says, and cast afterwards — which is both exact in f32 and, in bf16,
    /// closer to the f32 numbers than merging after the cast would be.
    dtype: DType,
    /// The base tensors that were read, so an adapter delta for one of them
    /// cannot go unnoticed.
    loaded: RefCell<BTreeSet<String>>,
}

struct Adapter {
    tensors: candle_core::safetensors::MmapedSafetensors,
    /// `lora_alpha / r`, the scale peft folds into the delta.
    scale: f64,
    /// Every tensor the file holds, so that nothing in it can be passed over
    /// without saying so.
    all: BTreeSet<String>,
    /// The ones the merge actually used.
    used: RefCell<BTreeSet<String>>,
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
    pub fn open(
        base: &Path,
        adapter: Option<&Path>,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
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
                let tensors = unsafe {
                    candle_core::safetensors::MmapedSafetensors::new(
                        dir.join("adapter_model.safetensors"),
                    )?
                };
                let all = tensors
                    .tensors()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                Some(Adapter {
                    tensors,
                    scale: config.lora_alpha / divisor,
                    all,
                    used: RefCell::new(BTreeSet::new()),
                })
            }
        };

        Ok(Self {
            base,
            adapter,
            device: device.clone(),
            dtype,
            loaded: RefCell::new(BTreeSet::new()),
        })
    }

    /// A tensor the adapter never touches (the embeddings, the norms, the
    /// convolution and the per-head scalars), in the backbone's dtype.
    pub fn plain(&self, path: &str) -> Result<Tensor> {
        Ok(self.plain_f32(path)?.to_dtype(self.dtype)?)
    }

    /// The same, kept in f32 — for the few weights the reference computes with
    /// in f32 whatever the backbone runs in.
    pub fn plain_f32(&self, path: &str) -> Result<Tensor> {
        let name = format!("model.{path}");
        let tensor = self.base.load(&name, &self.device)?.to_dtype(DType::F32)?;
        // Which base tensors the backbone actually runs: an adapter delta for one
        // of these that the merge never applied changes the answers, and an
        // adapter delta for anything else cannot.
        self.loaded.borrow_mut().insert(name);
        Ok(tensor)
    }

    /// A norm weight, which is stored as the deviation from 1.0 in the Qwen3.5
    /// bases (`Qwen3_5RMSNorm` computes `x * (1 + weight)` from a zero-centred
    /// parameter). Folding the 1.0 in here keeps the forward pass to one
    /// multiplication, as candle's `rms_norm` expects.
    pub fn zero_centred_norm(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain_f32(&format!("{path}.weight"))?;
        Ok((weight + 1.0)?.to_dtype(self.dtype)?)
    }

    /// A projection, with the adapter's `B @ A` delta merged in — in f32, before
    /// the cast to the backbone's dtype, as `LoadOptions.merge` does it.
    pub fn adapted(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain_f32(&format!("{path}.weight"))?;
        let Some(adapter) = &self.adapter else {
            return Ok(weight.to_dtype(self.dtype)?);
        };
        let Some((a, b)) = self.lora(adapter, path)? else {
            return Ok(weight.to_dtype(self.dtype)?);
        };
        let delta = (b.matmul(&a)? * adapter.scale)?;
        if delta.dims() != weight.dims() {
            return Err(Error::Engine(format!(
                "the adapter's delta for {path} is {:?}, the weight is {:?}",
                delta.dims(),
                weight.dims()
            )));
        }
        Ok((weight + delta)?.to_dtype(self.dtype)?)
    }

    /// Refuse a checkpoint whose adapter holds anything the merge passed over.
    ///
    /// This is the quietest way to serve the wrong model: a LoRA pair the merge
    /// never looked up leaves the weights loading, every answer looking
    /// reasonable, and the numbers somebody else's. peft names the module it
    /// wrapped `base_model.model.<path>`, and [`Weights::lora`] tries the two
    /// prefixes kev's own wrapping produces — if a checkpoint ever names them
    /// otherwise, or adapts a module this backbone merges nothing into, this says
    /// so by name instead of letting the answers drift.
    ///
    /// A delta for something the backbone never reads at all (a vocabulary head,
    /// say — Kev's answers do not come from one) cannot change an answer, so that
    /// is a warning rather than a refusal.
    pub(crate) fn adapter_fully_merged(&self) -> Result<()> {
        let Some(adapter) = &self.adapter else {
            return Ok(());
        };
        let used = adapter.used.borrow();
        let loaded = self.loaded.borrow();

        let mut ignored = Vec::new();
        let mut unreachable = Vec::new();
        for name in adapter.all.difference(&used) {
            match target_of(name) {
                // A weight this backbone runs, whose delta was never applied.
                Some(target) if loaded.contains(&target) => ignored.push(name.clone()),
                // A module it does not run; the reference's answers cannot depend
                // on it either, since they come from the hidden states.
                Some(_) => unreachable.push(name.clone()),
                // Not a LoRA pair at all, or under a prefix nothing here knows:
                // an adapter that does something this crate does not implement.
                None => ignored.push(name.clone()),
            }
        }

        if !unreachable.is_empty() {
            eprintln!(
                "warning: the adapter carries {} tensor(s) for modules this backbone \
                 does not run, so they cannot reach an answer: {}",
                unreachable.len(),
                shorten(&unreachable)
            );
        }
        if !ignored.is_empty() {
            let message = format!(
                "the adapter holds {} tensor(s) the merge never applied, which would \
                 serve a different model than the checkpoint describes: {}. Either \
                 this crate does not implement what the adapter does, or it names its \
                 modules differently than peft's `base_model.model.<path>`. Set \
                 {ALLOW_UNUSED}=1 to load anyway, once you are satisfied those \
                 tensors cannot change an answer.",
                ignored.len(),
                shorten(&ignored)
            );
            if !unused_tensors_allowed() {
                return Err(Error::Engine(message));
            }
            eprintln!("warning: {message}");
        }
        Ok(())
    }

    fn lora(&self, adapter: &Adapter, path: &str) -> Result<Option<(Tensor, Tensor)>> {
        // peft names the module it wrapped `base_model.model.<path>`; kev wraps
        // the text model, so `<path>` is what the base file calls `model.<path>`.
        for prefix in ["base_model.model", "base_model.model.model"] {
            let a = format!("{prefix}.{path}.lora_A.weight");
            let b = format!("{prefix}.{path}.lora_B.weight");
            if let Ok(first) = adapter.tensors.load(&a, &self.device) {
                let second = adapter.tensors.load(&b, &self.device)?;
                let mut used = adapter.used.borrow_mut();
                used.insert(a);
                used.insert(b);
                return Ok(Some((
                    first.to_dtype(DType::F32)?,
                    second.to_dtype(DType::F32)?,
                )));
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
pub(crate) fn attention_mask<F>(
    len: usize,
    allowed: F,
    device: &Device,
    dtype: DType,
) -> Result<Tensor>
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
    Ok(Tensor::from_vec(values, (1, 1, len, len), device)?.to_dtype(dtype)?)
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
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let inverse: Vec<f32> = (0..dim / 2)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let inverse = Tensor::from_vec(inverse, (1, dim / 2), device)?;
    let angles: Vec<f32> = positions.iter().map(|p| *p as f32).collect();
    let angles = Tensor::from_vec(angles, (positions.len(), 1), device)?;
    let angles = angles.matmul(&inverse)?;
    // The tables are built in f32 whatever the backbone runs in: a position's
    // angle is not something to round early.
    Ok((
        angles.cos()?.to_dtype(dtype)?,
        angles.sin()?.to_dtype(dtype)?,
    ))
}

/// Apply the rotary embedding to the first `2 * cos.dim(1)` components of each
/// head, leaving the rest as they are.
///
/// Qwen3 rotates the whole head; Qwen3.5 rotates a quarter of it
/// (`partial_rotary_factor`) and passes the remainder through, as
/// `apply_rotary_pos_emb` does by slicing at `cos.shape[-1]`.
pub(crate) fn rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let rotary = 2 * cos.dim(candle_core::D::Minus1)?;
    let dim = xs.dim(3)?;
    let xs = xs.contiguous()?;
    if rotary == dim {
        return candle_nn::rotary_emb::rope(&xs, cos, sin);
    }
    let rotated = candle_nn::rotary_emb::rope(&xs.narrow(3, 0, rotary)?.contiguous()?, cos, sin)?;
    // The concatenation can come back as a view, and a matmul wants neither of
    // its sides strided.
    Tensor::cat(&[rotated, xs.narrow(3, rotary, dim - rotary)?], 3)?.contiguous()
}

/// The additive mask for a batch of branches continuing from one prefilled
/// state, `[rows, 1, padded, state + padded]`.
///
/// Rows are padded to the longest; a pad key is closed to everyone, and a pad
/// query is left the state to look at so that no row of the softmax is empty.
pub(crate) fn branch_batch_mask(
    state: usize,
    lengths: &[usize],
    padded: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(lengths.len() * padded * (state + padded));
    for length in lengths {
        for query in 0..padded {
            for key in 0..state + padded {
                let allowed = if key < state {
                    true
                } else {
                    let key = key - state;
                    key < *length && query < *length && key <= query
                };
                values.push(if allowed { 0.0 } else { f32::MIN });
            }
        }
    }
    Ok(
        Tensor::from_vec(values, (lengths.len(), 1, padded, state + padded), device)?
            .to_dtype(dtype)?,
    )
}

/// Token ids and position ids for a batch of rows, padded to the longest with
/// zeros. A pad token is a real token id as far as the model is concerned; what
/// keeps it out of the answers is the mask and the readout.
pub(crate) fn pad_rows(rows: &[(&[u32], &[u32])]) -> (Vec<u32>, Vec<u32>, Vec<usize>, usize) {
    let padded = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
    let mut ids = Vec::with_capacity(rows.len() * padded);
    let mut positions = Vec::with_capacity(rows.len() * padded);
    let lengths = rows.iter().map(|(ids, _)| ids.len()).collect();
    for (row, row_positions) in rows {
        ids.extend_from_slice(row);
        ids.resize(ids.len() + padded - row.len(), 0);
        positions.extend_from_slice(row_positions);
        positions.resize(positions.len() + padded - row_positions.len(), 0);
    }
    (ids, positions, lengths, padded)
}

/// The additive mask for a batch of states prefilled together,
/// `[rows, 1, padded, padded]`: causal within each row's own length.
///
/// A pad key is closed to every real query. A pad query keeps its diagonal, so
/// that no row of the softmax is empty — what it computes is discarded.
pub(crate) fn prefill_batch_mask(
    lengths: &[usize],
    padded: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(lengths.len() * padded * padded);
    for length in lengths {
        for query in 0..padded {
            for key in 0..padded {
                let allowed = if query < *length {
                    key <= query && key < *length
                } else {
                    key == query
                };
                values.push(if allowed { 0.0 } else { f32::MIN });
            }
        }
    }
    Ok(Tensor::from_vec(values, (lengths.len(), 1, padded, padded), device)?.to_dtype(dtype)?)
}

/// `1.0` at a row's real tokens and `0.0` at its padding, `[rows, padded]`.
///
/// A recurrence has no mask to hide padding behind: it walks the tokens. Zeroing
/// the decay and the write strength there is what makes a padded row hand on the
/// state it had at its last real token — decay `0` means `exp(0) = 1`, and a
/// write strength of `0` writes nothing.
pub(crate) fn real_mask(lengths: &[usize], padded: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(lengths.len() * padded);
    for length in lengths {
        for token in 0..padded {
            values.push(if token < *length { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(values, (lengths.len(), padded), device)?)
}

/// The base tensor a leftover adapter tensor would have been merged into, or
/// `None` when it is not a LoRA pair under a prefix this crate knows.
fn target_of(name: &str) -> Option<String> {
    let stem = name
        .strip_suffix(".lora_A.weight")
        .or_else(|| name.strip_suffix(".lora_B.weight"))?;
    // The same two prefixes `Weights::lora` looks under, longest first.
    for prefix in ["base_model.model.model.", "base_model.model."] {
        if let Some(path) = stem.strip_prefix(prefix) {
            return Some(format!("model.{path}.weight"));
        }
    }
    None
}

/// A few names, and then the count — enough to act on, short enough to read.
fn shorten(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(4).map(String::as_str).collect();
    if names.len() <= shown.len() {
        return shown.join(", ");
    }
    format!(
        "{}, and {} more",
        shown.join(", "),
        names.len() - shown.len()
    )
}
