//! A [`Forward`] backend over the candle Qwen3 backbone.
//!
//! This is the half a checkpoint ships as weights: the base model with the
//! adapter merged ([`Backbone`]), its tokenizer, and the pointer head. Put them
//! together and [`LocalEngine`](crate::LocalEngine) answers System One requests
//! with no server in sight.

use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::error::{Error, Result};
use crate::local::{Forward, Pass};
use crate::qwen3::{attention_mask, Backbone};
use crate::readout::{Linear, PointerHead};

/// A loaded Qwen3 checkpoint, ready to answer [`Forward`] calls.
pub struct Qwen3Backend {
    model: Backbone,
    tokenizer: Tokenizer,
    device: Device,
}

impl Qwen3Backend {
    /// Load a base model and, optionally, the Kev adapter over it.
    ///
    /// `base` holds `config.json` and the base weights; `adapter` is the
    /// checkpoint directory (`adapter_config.json`,
    /// `adapter_model.safetensors`, and usually the tokenizer). The tokenizer
    /// is taken from the checkpoint when it has one, since that is the one it
    /// was trained with.
    pub fn open(base: &Path, adapter: Option<&Path>) -> Result<Self> {
        Self::open_on(base, adapter, Device::Cpu)
    }

    /// As [`Qwen3Backend::open`], on a device of your choosing.
    pub fn open_on(base: &Path, adapter: Option<&Path>, device: Device) -> Result<Self> {
        let path = tokenizer_path(base, adapter)?;
        let mut tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| Error::Engine(format!("cannot read {}: {e}", path.display())))?;

        // What `tok(text, add_special_tokens=False)` amounts to on the Python
        // side (transformers' `TokenizersBackend._encode_plus`): truncation and
        // padding are turned off on every call, and `encode_special_tokens`
        // stays false, so a special token appearing in the text is matched
        // rather than split. Two of those we would otherwise inherit from
        // `tokenizer.json`, which is a difference in the token ids, so say all
        // three out loud.
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::Engine(format!("cannot disable truncation: {e}")))?;
        tokenizer.with_padding(None);
        tokenizer.set_encode_special_tokens(false);
        Ok(Self {
            model: Backbone::load(base, adapter, &device)?,
            tokenizer,
            device,
        })
    }

    /// The width of the hidden states this backbone produces; the pointer head
    /// has to expect the same.
    pub fn hidden_size(&self) -> usize {
        self.model.hidden_size()
    }
}

impl std::fmt::Debug for Qwen3Backend {
    // A loaded model is an opaque handle; its width is the only useful fact.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Backend")
            .field("hidden_size", &self.model.hidden_size())
            .finish_non_exhaustive()
    }
}

fn tokenizer_path(base: &Path, adapter: Option<&Path>) -> Result<PathBuf> {
    let candidates = adapter
        .into_iter()
        .chain(std::iter::once(base))
        .map(|dir| dir.join("tokenizer.json"));
    for path in candidates {
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error::Engine(format!(
        "no tokenizer.json in the checkpoint or in {}",
        base.display()
    )))
}

impl Forward for Qwen3Backend {
    fn tokenise(&mut self, text: &str) -> Result<Vec<u32>> {
        // Never with special tokens: the delimiters are added by the layout,
        // and caller text has been escaped so it cannot produce one.
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Error::Engine(format!("cannot tokenise: {e}")))?;
        Ok(encoded.get_ids().to_vec())
    }

    fn delimiter(&mut self, token: &str) -> Result<u32> {
        self.tokenizer.token_to_id(token).ok_or_else(|| {
            Error::Engine(format!(
                "this tokenizer has no {token}, so it is not a Qwen tokenizer Kev can use"
            ))
        })
    }

    fn hidden(&mut self, pass: &Pass<'_>) -> Result<Vec<Vec<f32>>> {
        let mask = attention_mask(
            pass.ids.len(),
            |query, key| pass.attends(query, key),
            &self.device,
        )?;
        let hidden = self.model.forward(pass.ids, pass.positions, &mask)?;

        let wanted: Vec<u32> = pass.readout.iter().map(|index| *index as u32).collect();
        let wanted = Tensor::from_vec(wanted, pass.readout.len(), &self.device)?;
        Ok(hidden.index_select(&wanted, 0)?.to_vec2::<f32>()?)
    }
}

/// The checkpoint's pointer head: two projections and the calibration.
///
/// Reads `head.pt` (what a Kev run directory ships, a torch pickle) or a
/// safetensors file with the same tensors. The temperature is passed in
/// because it is a scalar rather than a tensor, and tensor files are all these
/// readers return: a checkpoint's own value is in `head.pt`'s `temperature`
/// field, `/v1/models` reports it, and `1.0` means the raw head.
pub fn pointer_head(path: &Path, temperature: f32) -> Result<PointerHead> {
    let tensors = if path.extension().is_some_and(|e| e == "safetensors") {
        let file = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        file.tensors()
            .into_iter()
            .map(|(name, _)| {
                let tensor = file.load(&name, &Device::Cpu)?;
                Ok((name, tensor))
            })
            .collect::<candle_core::Result<Vec<_>>>()?
    } else {
        candle_core::pickle::read_all(path)?
    };

    let find = |suffix: &str| -> Result<Tensor> {
        tensors
            .iter()
            .find(|(name, _)| name == suffix || name.ends_with(&format!(".{suffix}")))
            .map(|(_, tensor)| tensor.clone())
            .ok_or_else(|| {
                Error::Engine(format!(
                    "{} has no {suffix}; it holds {:?}",
                    path.display(),
                    tensors.iter().map(|(name, _)| name).collect::<Vec<_>>()
                ))
            })
    };

    let projection = |name: &str| -> Result<Linear> {
        let weight = find(&format!("{name}.weight"))?.to_dtype(candle_core::DType::F32)?;
        let bias = find(&format!("{name}.bias"))?.to_dtype(candle_core::DType::F32)?;
        let (_, inputs) = weight.dims2()?;
        Linear::new(weight.flatten_all()?.to_vec1()?, bias.to_vec1()?, inputs)
    };

    PointerHead::new(projection("q")?, projection("k")?)?.with_temperature(temperature)
}
