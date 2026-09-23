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
/// Reads `head.pt` — what a Kev run directory ships, a torch pickle — including
/// the temperature it was calibrated with, or a safetensors file with the same
/// tensors, which carries no temperature and so leaves the head raw.
///
/// For raw logits from a calibrated checkpoint, override it afterwards:
/// `pointer_head(path)?.with_temperature(1.0)?`.
pub fn pointer_head(path: &Path) -> Result<PointerHead> {
    let is_safetensors = path.extension().is_some_and(|e| e == "safetensors");
    let tensors = if is_safetensors {
        let file = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
        file.tensors()
            .into_iter()
            .map(|(name, _)| {
                let tensor = file.load(&name, &Device::Cpu)?;
                Ok((name, tensor))
            })
            .collect::<candle_core::Result<Vec<_>>>()?
    } else {
        // kev's head.pt is `Meta`: a dict of metadata with the head's state dict
        // under `head`. Read without that key, the reader walks the metadata
        // instead and finds no tensors at all. A bare state dict still works.
        match candle_core::pickle::read_all_with_key(path, Some("head")) {
            Ok(tensors) if !tensors.is_empty() => tensors,
            _ => candle_core::pickle::read_all(path)?,
        }
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

    let head = PointerHead::new(projection("q")?, projection("k")?)?;
    match if is_safetensors {
        None
    } else {
        temperature(path)?
    } {
        Some(temperature) => head.with_temperature(temperature),
        // `Meta.temperature` defaults to 1.0 for checkpoints that never had one
        // fitted, and 1.0 is the raw head.
        None => Ok(head),
    }
}

/// The calibration temperature a checkpoint carries in `head.pt`, if it has one.
///
/// `torch.save` writes a zip holding a pickled dict (`kev.checkpoint.Meta`), and
/// the temperature is one of its values. Tensor readers skip it because it is a
/// scalar, so walk the pickle for it: the value is load-bearing, since the head
/// divides its scores by it. It never changes which option wins, only how
/// confident the distribution looks — which is exactly the kind of difference
/// that goes unnoticed.
pub fn temperature(path: &Path) -> Result<Option<f32>> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::Engine(format!("cannot read {}: {e}", path.display())))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| Error::Engine(format!("{} is not a torch archive: {e}", path.display())))?;
    let pickled = zip
        .file_names()
        .find(|name| name.ends_with("data.pkl"))
        .ok_or_else(|| Error::Engine(format!("{} holds no data.pkl", path.display())))?
        .to_string();

    let reader = zip
        .by_name(&pickled)
        .map_err(|e| Error::Engine(format!("cannot read {pickled}: {e}")))?;
    let mut stack = candle_core::pickle::Stack::empty();
    stack.read_loop(&mut std::io::BufReader::new(reader))?;
    Ok(scalar(&stack.finalize()?, "temperature").map(|value| value as f32))
}

/// The number stored under `key`, anywhere in a pickled object. Shallower
/// matches win, so a checkpoint's own `temperature` beats one nested in
/// whatever else it saved.
fn scalar(object: &candle_core::pickle::Object, key: &str) -> Option<f64> {
    use candle_core::pickle::Object;

    let number = |object: &Object| match object {
        Object::Float(value) => Some(*value),
        Object::Int(value) => Some(f64::from(*value)),
        Object::Long(value) => Some(*value as f64),
        _ => None,
    };
    match object {
        Object::Dict(entries) => entries
            .iter()
            .find_map(|(name, value)| match name {
                Object::Unicode(name) if name == key => number(value),
                _ => None,
            })
            .or_else(|| entries.iter().find_map(|(_, value)| scalar(value, key))),
        Object::Tuple(items) | Object::List(items) => {
            items.iter().find_map(|item| scalar(item, key))
        }
        Object::Reduce { args, .. } | Object::Build { args, .. } => scalar(args, key),
        Object::PersistentLoad(inner) => scalar(inner, key),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A protocol-2 pickle, which is what `torch.save` writes by default:
    /// `{"base": "...", "head": {"dp": 256}, "temperature": 2.34, "lora": 16}`.
    /// Hand-assembled so the walk can be tested without torch.
    fn pickled_meta() -> Vec<u8> {
        let mut bytes = vec![0x80, 0x02, b'}', b'q', 0x00, b'('];
        let text = |bytes: &mut Vec<u8>, value: &str| {
            bytes.push(b'X');
            bytes.extend((value.len() as u32).to_le_bytes());
            bytes.extend(value.as_bytes());
        };
        text(&mut bytes, "base");
        text(&mut bytes, "Qwen/Qwen3-4B-Base");
        // A nested dict the walk has to step over, holding a number of its own.
        text(&mut bytes, "head");
        bytes.extend(*b"}(");
        text(&mut bytes, "dp");
        bytes.push(b'J');
        bytes.extend(256i32.to_le_bytes());
        bytes.push(b'u');
        text(&mut bytes, "temperature");
        bytes.push(b'G');
        bytes.extend(2.34f64.to_be_bytes()); // pickle writes floats big-endian
        text(&mut bytes, "lora");
        bytes.push(b'J');
        bytes.extend(16i32.to_le_bytes());
        bytes.extend(*b"u.");
        bytes
    }

    fn unpickle(bytes: &[u8]) -> candle_core::pickle::Object {
        let mut stack = candle_core::pickle::Stack::empty();
        stack
            .read_loop(&mut std::io::BufReader::new(bytes))
            .unwrap();
        stack.finalize().unwrap()
    }

    #[test]
    fn the_temperature_is_read_out_of_a_torch_saved_dict() {
        let meta = unpickle(&pickled_meta());

        assert_eq!(scalar(&meta, "temperature"), Some(2.34));
        // Integers count: a checkpoint could have saved 1 rather than 1.0.
        assert_eq!(scalar(&meta, "lora"), Some(16.0));
        // Nested values are found, but only after the top level.
        assert_eq!(scalar(&meta, "dp"), Some(256.0));
        // A checkpoint without one leaves the head raw rather than guessing.
        assert_eq!(scalar(&meta, "holdout"), None);
        assert_eq!(scalar(&meta, "base"), None, "a string is not a number");
    }

    #[test]
    fn the_temperature_comes_out_of_a_head_pt_file() {
        // A torch archive is a zip with the pickle at <name>/data.pkl. A real
        // one holds the head's tensors beside it, in data/0 and friends, which
        // the temperature does not depend on.
        let path = std::env::temp_dir().join("kev-head-temperature.pt");
        let file = std::fs::File::create(&path).unwrap();
        let mut archive = zip::ZipWriter::new(file);
        archive
            .start_file("archive/data.pkl", zip::write::SimpleFileOptions::default())
            .unwrap();
        std::io::Write::write_all(&mut archive, &pickled_meta()).unwrap();
        archive.finish().unwrap();

        assert_eq!(temperature(&path).unwrap(), Some(2.34));
        std::fs::remove_file(&path).ok();
    }

    /// One `torch._utils._rebuild_tensor_v2` call, as protocol 2 pickles it:
    /// the storage as a persistent id, then offset, size and stride.
    fn pickled_tensor(bytes: &mut Vec<u8>, storage: &str, rows: usize, columns: usize) {
        let global = |bytes: &mut Vec<u8>, module: &str, name: &str| {
            bytes.push(b'c');
            bytes.extend(format!("{module}\n{name}\n").as_bytes());
        };
        let text = |bytes: &mut Vec<u8>, value: &str| {
            bytes.push(b'X');
            bytes.extend((value.len() as u32).to_le_bytes());
            bytes.extend(value.as_bytes());
        };
        let int = |bytes: &mut Vec<u8>, value: i32| {
            bytes.push(b'J');
            bytes.extend(value.to_le_bytes());
        };

        global(bytes, "torch._utils", "_rebuild_tensor_v2");
        bytes.push(b'(');
        // The storage: ("storage", torch.FloatStorage, "<file>", "cpu", numel)
        bytes.push(b'(');
        text(bytes, "storage");
        global(bytes, "torch", "FloatStorage");
        text(bytes, storage);
        text(bytes, "cpu");
        int(bytes, (rows * columns.max(1)) as i32);
        bytes.extend(*b"tQ");
        int(bytes, 0); // storage offset
        bytes.push(b'(');
        int(bytes, rows as i32);
        if columns > 0 {
            int(bytes, columns as i32);
        }
        bytes.push(b't');
        bytes.push(b'(');
        int(bytes, columns.max(1) as i32);
        if columns > 0 {
            int(bytes, 1);
        }
        bytes.push(b't');
        bytes.push(0x88); // requires_grad
        bytes.extend(*b"tR");
    }

    /// A `head.pt` the way `kev.checkpoint.write_meta` saves one: the metadata
    /// dict, the head's state dict under `head` as an OrderedDict, and one zip
    /// entry per storage.
    fn write_head_pt(path: &Path, temperature: f64, tensors: &[(&str, Vec<f32>, usize, usize)]) {
        let text = |bytes: &mut Vec<u8>, value: &str| {
            bytes.push(b'X');
            bytes.extend((value.len() as u32).to_le_bytes());
            bytes.extend(value.as_bytes());
        };

        let mut pickle = vec![0x80, 0x02, b'}', b'q', 0x00, b'('];
        text(&mut pickle, "base");
        text(&mut pickle, "Qwen/Qwen3-4B-Base");
        text(&mut pickle, "temperature");
        pickle.push(b'G');
        pickle.extend(temperature.to_be_bytes());
        text(&mut pickle, "head");
        // An OrderedDict, which is what a torch state_dict pickles as.
        pickle.push(b'c');
        pickle.extend(b"collections\nOrderedDict\n");
        pickle.extend(*b")R(");
        for (index, (name, _, rows, columns)) in tensors.iter().enumerate() {
            text(&mut pickle, name);
            pickled_tensor(&mut pickle, &index.to_string(), *rows, *columns);
        }
        pickle.extend(*b"uu.");

        let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        let stored = zip::write::SimpleFileOptions::default();
        archive.start_file("archive/data.pkl", stored).unwrap();
        std::io::Write::write_all(&mut archive, &pickle).unwrap();
        for (index, (_, values, _, _)) in tensors.iter().enumerate() {
            archive
                .start_file(format!("archive/data/{index}"), stored)
                .unwrap();
            for value in values {
                std::io::Write::write_all(&mut archive, &value.to_le_bytes()).unwrap();
            }
        }
        archive.finish().unwrap();
    }

    #[test]
    fn a_pointer_head_is_loaded_from_head_pt_with_its_calibration() {
        // The whole file: the two projections out of the state dict under `head`,
        // and the temperature out of the metadata beside it.
        let (hidden, pointer) = (4usize, 3usize);
        let weight: Vec<f32> = (0..pointer * hidden)
            .map(|i| (i as f32 - 5.0) / 4.0)
            .collect();
        let bias: Vec<f32> = (0..pointer).map(|i| i as f32 / 10.0).collect();
        let path = std::env::temp_dir().join("kev-head-with-calibration.pt");
        write_head_pt(
            &path,
            2.34,
            &[
                ("q.weight", weight.clone(), pointer, hidden),
                ("q.bias", bias.clone(), pointer, 0),
                ("k.weight", weight.clone(), pointer, hidden),
                ("k.bias", bias.clone(), pointer, 0),
            ],
        );

        let loaded = pointer_head(&path).unwrap();
        let expected = PointerHead::new(
            Linear::new(weight.clone(), bias.clone(), hidden).unwrap(),
            Linear::new(weight, bias, hidden).unwrap(),
        )
        .unwrap();

        let decide = vec![0.3, -0.7, 1.1, 0.5];
        let options = vec![vec![1.0, 0.2, -0.4, 0.8], vec![-0.6, 0.9, 0.1, -0.3]];
        let raw = expected.logits(&decide, &options).unwrap();
        let calibrated = loaded.logits(&decide, &options).unwrap();

        assert_eq!(raw.len(), 2);
        for (raw, calibrated) in raw.iter().zip(&calibrated) {
            // Same weights, and the checkpoint's temperature divided in.
            assert!(
                (raw / 2.34 - calibrated).abs() < 1e-5,
                "{raw} / 2.34 vs {calibrated}"
            );
            assert!(raw.abs() > 1e-3, "the test weights produce no signal");
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_file_that_is_not_a_torch_archive_says_so() {
        let path = std::env::temp_dir().join("kev-head-not-an-archive.pt");
        std::fs::write(&path, b"this is not a zip").unwrap();

        let error = temperature(&path).unwrap_err();

        assert!(error.to_string().contains("torch archive"), "{error}");
        std::fs::remove_file(&path).ok();
    }
}
