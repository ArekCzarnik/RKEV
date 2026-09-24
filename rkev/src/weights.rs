//! Reading a checkpoint's tensors, with its LoRA adapter merged in.
//!
//! Shared by the backbones: a Kev checkpoint is always a base model plus a
//! rank-16 adapter, and merging it at load time in f32 is what the Python does
//! (`LoadOptions.merge`), exactly there.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::Module;
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

/// How the projections are stored, for the checkpoints that are too big to serve
/// dense.
///
/// Quantising happens **after** the LoRA merge, never before: the merge is exact
/// in f32 (`Weights::merged`), and only the result is rounded down to blocks. A
/// pre-quantised base model could not be merged into at all.
///
/// The four that are worth having on a model this small. Bits per weight are what
/// they cost in bandwidth, which is what a CPU pass is bound by: f32 is 32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantisation {
    /// 4.5 bits. The usual choice, and the usual place to lose accuracy.
    Q4K,
    /// 5.5 bits.
    Q5K,
    /// 6.6 bits. Close to lossless on most weights.
    Q6K,
    /// 8.5 bits, block-wise scaled. The conservative one.
    Q8_0,
}

impl Quantisation {
    /// By name, as a caller would type it: `q4k`, `q5k`, `q6k`, `q8_0`.
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace('-', "_").as_str() {
            "q4k" | "q4_k" => Ok(Self::Q4K),
            "q5k" | "q5_k" => Ok(Self::Q5K),
            "q6k" | "q6_k" => Ok(Self::Q6K),
            "q8_0" | "q80" | "q8" => Ok(Self::Q8_0),
            other => Err(Error::Engine(format!(
                "unknown quantisation {other:?}; this crate offers q4k, q5k, q6k and q8_0"
            ))),
        }
    }

    fn ggml(self) -> GgmlDType {
        match self {
            Self::Q4K => GgmlDType::Q4K,
            Self::Q5K => GgmlDType::Q5K,
            Self::Q6K => GgmlDType::Q6K,
            Self::Q8_0 => GgmlDType::Q8_0,
        }
    }
}

/// One projection of the backbone: a merged weight, dense or quantised.
pub(crate) enum Projection {
    Dense(Tensor),
    Quantised(QMatMul),
}

impl Projection {
    /// `xs @ weight^T`, the same arithmetic either way — the quantised form
    /// dequantises a block at a time inside the kernel rather than up front.
    pub(crate) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(weight) => linear(xs, weight),
            Self::Quantised(matmul) => matmul.forward(xs),
        }
    }
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
    /// Set when the projections are to be quantised after merging.
    quantise: Option<Quantisation>,
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
        quantise: Option<Quantisation>,
    ) -> Result<Self> {
        if quantise.is_some() {
            // The ggml kernels take f32 or f16 activations; f32 is what the
            // quantised path is worth measuring against, and mixing a reduced
            // activation precision into a reduced weight precision would make the
            // two losses impossible to tell apart.
            if dtype != DType::F32 {
                return Err(Error::Engine(format!(
                    "quantised projections run with f32 activations, not {dtype:?}"
                )));
            }
            // `QTensor::quantize` is a CPU routine, and a quantised matmul on
            // another device is untried here.
            if !device.is_cpu() {
                return Err(Error::Engine(String::from(
                    "quantisation is implemented for the CPU only",
                )));
            }
        }
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
            quantise,
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

    /// A projection as the backbone will use it: dense in its own dtype, or
    /// quantised when the caller asked for that.
    ///
    /// The adapter's `B @ A` delta is merged in f32 first, before any cast and
    /// before any quantisation, as `LoadOptions.merge` does it.
    pub fn projection(&self, path: &str) -> Result<Projection> {
        let merged = self.merged(path)?;
        match self.quantise {
            None => Ok(Projection::Dense(merged.to_dtype(self.dtype)?)),
            Some(quantisation) => {
                // The k-quants pack 256 weights to a super-block and q8_0 packs
                // 32, and a row that does not divide evenly cannot be packed at
                // all. candle says so in terms of block sizes and dimensions;
                // this says which projection, and what else would fit.
                let block = quantisation.ggml().block_size();
                let row = *merged.dims().last().unwrap_or(&0);
                if row % block != 0 {
                    return Err(Error::Engine(format!(
                        "{path} is {:?}, and {quantisation:?} packs {block} weights to a \
                         block, which {row} does not divide by. q8_0 packs 32; below that \
                         the projection has to stay dense.",
                        merged.dims()
                    )));
                }
                let quantised = QTensor::quantize(&merged, quantisation.ggml())?;
                Ok(Projection::Quantised(QMatMul::from_qtensor(quantised)?))
            }
        }
    }

    /// The merged weight in f32, whatever the backbone runs in.
    fn merged(&self, path: &str) -> Result<Tensor> {
        let weight = self.plain_f32(&format!("{path}.weight"))?;
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
    let dims = xs.dims();
    if dims.len() < 3 {
        return xs.broadcast_matmul(&weight.t()?);
    }
    // Every leading dimension is folded into the rows, which is what the quantised
    // kernel does with its own input. `broadcast_matmul` would instead broadcast the
    // *weight* to the batch and materialise it — candle says so in a TODO — so a
    // 2048x1024 projection at five rows copies 40 MB per call. Per projection, per
    // layer, per pass: on a 0.6B model with five questions that is gigabytes of
    // memcpy, and it was the reason a branch pass cost three times a packed one.
    let (rows, inner) = (
        dims[..dims.len() - 1].iter().product::<usize>(),
        dims[dims.len() - 1],
    );
    let out = xs
        .contiguous()?
        .reshape((rows, inner))?
        .matmul(&weight.t()?)?;
    let mut shape = dims[..dims.len() - 1].to_vec();
    shape.push(out.dim(1)?);
    out.reshape(shape)
}

pub(crate) fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    candle_nn::ops::rms_norm(&xs.contiguous()?, weight, eps as f32)
}

/// `lhs @ rhs` where every row of the batch shares the same `rhs`:
/// `[batch, heads, rows, inner]` against `[1, heads, inner, cols]`.
///
/// The obvious spellings both copy the shared side once per row.
/// `broadcast_matmul` concretises the broadcast (candle says so in a TODO), and
/// expanding it by hand before `cat` does the same — which for a state's keys and
/// values is what a branch pass spends its time on: at 571 state tokens and 28
/// layers that is hundreds of megabytes of memcpy per request, five times over for
/// five questions.
///
/// Folding the batch into the row dimension instead copies only the small side: the
/// matmul becomes `[heads, batch * rows, inner] @ [heads, inner, cols]`, and the
/// state is read in place.
pub(crate) fn shared_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (batch, heads, rows, inner) = lhs.dims4()?;
    let (shared, rhs_heads, rhs_inner, cols) = rhs.dims4()?;
    if shared != 1 || rhs_heads != heads || rhs_inner != inner {
        return Err(Error::Engine(format!(
            "a shared matmul wants [1, {heads}, {inner}, cols] on the right, got {:?}",
            rhs.dims()
        )));
    }
    let folded = lhs
        .transpose(0, 1)?
        .contiguous()?
        .reshape((heads, batch * rows, inner))?;
    let shared = rhs.reshape((heads, inner, cols))?;
    Ok(folded
        .matmul(&shared)?
        .reshape((heads, batch, rows, cols))?
        .transpose(0, 1)?
        .contiguous()?)
}

/// Softmax attention over a batch of rows, grouped-query, optionally continuing
/// from a prefilled state: `[batch, heads, len, dim]` out.
///
/// `q` is `[batch, heads, len, dim]`, `k` and `v` are `[batch, kv_heads, len,
/// dim]`, and `mask` is additive over `[.., len, state + len]`. A state comes as
/// `(keys, values)` from [`state_keys`] and the prefill: `[1, kv_heads, dim,
/// state]` and `[1, kv_heads, state, dim]`.
///
/// Query head `h` reads key/value head `h / group`, so the `group` query heads
/// that share one are stacked along the rows and multiplied against it once. The
/// obvious spelling repeats every key/value head `group` times instead, which
/// copies them — and for a state that is shared by every row and every request
/// that hits the cache, it copied the whole state per layer per pass.
///
/// The state is scored separately from the branch rather than concatenated in
/// front of it, for the same reason: it is one tensor every row shares, and
/// [`shared_matmul`] reads it where it lies. A matmul distributes over the
/// concatenation, so this is the same arithmetic.
pub(crate) fn grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    past: Option<(&Tensor, &Tensor)>,
    mask: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (batch, heads, len, dim) = q.dims4()?;
    let kv_heads = k.dim(1)?;
    let group = heads / kv_heads;
    // `[batch, heads, len, cols]` <-> `[batch, kv_heads, group * len, cols]`: the
    // same memory, since the heads of one group are adjacent.
    let grouped = |x: &Tensor| -> Result<Tensor> {
        let cols = x.dim(3)?;
        Ok(x.contiguous()?
            .reshape((batch, kv_heads, group * len, cols))?)
    };
    let ungrouped = |x: Tensor| -> Result<Tensor> {
        let cols = x.dim(3)?;
        Ok(x.reshape((batch, heads, len, cols))?)
    };

    let queries = grouped(q)?;
    let branch = ungrouped(queries.matmul(&k.transpose(2, 3)?)?)?;
    let scores = match past {
        None => branch,
        Some((keys, _)) => Tensor::cat(&[ungrouped(shared_matmul(&queries, keys)?)?, branch], 3)?,
    };
    let scores = (scores * scale)?;
    // The mask is what keeps one question from reading another.
    let scores = scores.broadcast_add(mask)?;
    // The reference takes the softmax in f32 whatever the backbone runs in.
    let weights =
        candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;

    let attended = match past {
        None => grouped(&weights)?.matmul(v)?,
        Some((_, values)) => {
            let tokens = values.dim(2)?;
            let from_state = shared_matmul(&grouped(&weights.narrow(3, 0, tokens)?)?, values)?;
            let from_branch = grouped(&weights.narrow(3, tokens, len)?)?.matmul(v)?;
            (from_state + from_branch)?
        }
    };
    Ok(attended.reshape((batch, heads, len, dim))?)
}

/// A state's keys as [`grouped_attention`] reads them: `[.., kv_heads, dim,
/// tokens]`, transposed once at prefill rather than on every pass that
/// continues from it.
pub(crate) fn state_keys(keys: &Tensor) -> Result<Tensor> {
    Ok(keys.transpose(2, 3)?.contiguous()?)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Grouped-query attention the way it was spelled before [`grouped_attention`]:
    /// every key/value head repeated for the query heads that read it, the state's
    /// keys included — once per layer, per pass. Kept as the reference the grouped
    /// form is held to, and as the baseline it is measured against.
    fn repeated_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        past: Option<(&Tensor, &Tensor)>,
        mask: &Tensor,
        scale: f64,
    ) -> Result<Tensor> {
        let repeat = |xs: &Tensor| -> Result<Tensor> {
            let (batch, kv_heads, len, dim) = xs.dims4()?;
            let n = q.dim(1)? / kv_heads;
            Ok(xs
                .unsqueeze(2)?
                .expand((batch, kv_heads, n, len, dim))?
                .reshape((batch, kv_heads * n, len, dim))?)
        };
        let len = q.dim(2)?;
        let branch_keys = repeat(k)?;
        let scores = match past {
            None => q.matmul(&branch_keys.transpose(2, 3)?)?,
            Some((keys, _)) => {
                let state = repeat(keys)?.transpose(2, 3)?.contiguous()?;
                Tensor::cat(
                    &[
                        shared_matmul(q, &state)?,
                        q.matmul(&branch_keys.transpose(2, 3)?)?,
                    ],
                    3,
                )?
            }
        };
        let scores = (scores * scale)?.broadcast_add(mask)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let branch_values = repeat(v)?;
        Ok(match past {
            None => weights.matmul(&branch_values)?,
            Some((_, values)) => {
                let state = repeat(values)?;
                let tokens = state.dim(2)?;
                let from_state =
                    shared_matmul(&weights.narrow(3, 0, tokens)?.contiguous()?, &state)?;
                let from_branch = weights
                    .narrow(3, tokens, len)?
                    .contiguous()?
                    .matmul(&branch_values)?;
                (from_state + from_branch)?
            }
        })
    }

    /// Branches of `len` tokens continuing from a state of `state` tokens: the
    /// queries, the branch's keys and values, the state as a prefill keeps it
    /// (keys untransposed, for the repeated form to transpose), and the mask.
    struct Case {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        keys: Tensor,
        values: Tensor,
        mask: Tensor,
    }

    impl Case {
        fn new(
            batch: usize,
            heads: usize,
            kv_heads: usize,
            len: usize,
            state: usize,
            dim: usize,
        ) -> Self {
            let device = Device::Cpu;
            let noise = |shape: (usize, usize, usize, usize)| {
                Tensor::randn(0f32, 1f32, shape, &device).unwrap()
            };
            Self {
                q: noise((batch, heads, len, dim)),
                k: noise((batch, kv_heads, len, dim)),
                v: noise((batch, kv_heads, len, dim)),
                keys: noise((1, kv_heads, state, dim)),
                values: noise((1, kv_heads, state, dim)),
                mask: branch_batch_mask(state, &vec![len; batch], len, &device, DType::F32)
                    .unwrap(),
            }
        }

        fn grouped(&self) -> Tensor {
            let keys = state_keys(&self.keys).unwrap();
            let past = Some((&keys, &self.values));
            grouped_attention(&self.q, &self.k, &self.v, past, &self.mask, 0.125).unwrap()
        }

        fn repeated(&self) -> Tensor {
            let past = Some((&self.keys, &self.values));
            repeated_attention(&self.q, &self.k, &self.v, past, &self.mask, 0.125).unwrap()
        }
    }

    fn largest_difference(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn grouping_the_query_heads_is_the_same_attention_as_repeating_the_keys() {
        // Four query heads per key/value head, so a head read by the wrong group
        // cannot line up by accident; three rows, so the batch folding is in play.
        let case = Case::new(3, 8, 2, 5, 7, 4);

        let difference = largest_difference(&case.grouped(), &case.repeated());

        assert!(difference < 1e-5, "differs by {difference}");
    }

    /// What repeating the state's keys and values cost per layer, at kev-0.6b's
    /// shapes: sixteen query heads over eight key/value heads, 128 wide, a
    /// 571-token state and five branches of forty-five tokens.
    #[test]
    #[ignore = "a measurement, not an assertion: cargo test --release -- --ignored --nocapture"]
    fn what_reading_the_state_unrepeated_is_worth() {
        use std::time::Instant;

        let case = Case::new(5, 16, 8, 45, 571, 128);
        let rounds = 20;
        let time = |attend: &dyn Fn() -> Tensor| {
            attend(); // warm up
            let started = Instant::now();
            for _ in 0..rounds {
                attend();
            }
            started.elapsed().as_secs_f64() * 1000.0 / rounds as f64
        };

        let repeated = time(&|| case.repeated());
        // The grouped form transposes the state's keys once, at prefill; timed
        // here per pass anyway, so the comparison does not flatter it.
        let grouped = time(&|| case.grouped());
        let difference = largest_difference(&case.grouped(), &case.repeated());

        println!(
            "one attention layer, 5x45 branch rows over a 571-token state:\n  \
             repeated {repeated:>8.2} ms\n  grouped  {grouped:>8.2} ms ({:.1}x, largest \
             difference {difference:.1e})",
            repeated / grouped
        );
    }
}
