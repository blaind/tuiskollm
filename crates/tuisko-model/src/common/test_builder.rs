//! Synthetic safetensors fixture construction shared by the split target test modules.
//!
//! The builder owns only fixture *construction*: it appends tensors in call order, derives each
//! `data_offsets` span from the payload it has already written, and emits the same header and
//! payload the handwritten generators emitted. Expected values and assertions stay with the tests.

use crate::DType;
use serde_json::{Map, Value, json};
use std::fs;
use std::path::Path;

/// Incrementally assembled synthetic safetensors source.
pub(crate) struct SafeTensorTestBuilder {
    header: Map<String, Value>,
    payload: Vec<u8>,
}

impl SafeTensorTestBuilder {
    /// Empty source with no declared tensors.
    pub(crate) fn new() -> Self {
        Self {
            header: Map::new(),
            payload: Vec::new(),
        }
    }

    /// Appends `dtype` elements of `shape`, every source byte set to `fill`.
    pub(crate) fn add_raw(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: &[usize],
        fill: u8,
    ) -> &mut Self {
        let elements = shape.iter().copied().product::<usize>();
        let bytes = elements * dtype.byte_width() as usize;
        let begin = self.payload.len();

        self.payload.resize(begin + bytes, fill);
        self.declare(name, dtype, shape, begin)
    }

    /// Appends a BF16 tensor whose every element carries the exact source `word`.
    pub(crate) fn add_bf16(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        word: u16,
    ) -> &mut Self {
        let elements = shape.iter().copied().product::<usize>();
        let begin = self.payload.len();

        for _ in 0..elements {
            self.payload.extend_from_slice(&word.to_le_bytes());
        }

        self.declare(name, DType::Bf16, shape, begin)
    }

    /// Appends a BF16 tensor whose every element carries this tensor's own 1-based declaration
    /// ordinal as its source word, so a fixture's tensors stay distinguishable by value.
    pub(crate) fn add_bf16_ordinal(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
    ) -> &mut Self {
        let word = u16::try_from(self.header.len() + 1).unwrap();

        self.add_bf16(name, shape, word)
    }

    /// Appends a rank-0 F32 scalar carrying the exact source bits of `value`.
    pub(crate) fn add_rank0_f32(&mut self, name: impl Into<String>, value: f32) -> &mut Self {
        self.add_f32(name, &[], value)
    }

    /// Appends an F32 tensor whose every element carries the exact source bits of `value`.
    pub(crate) fn add_f32(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        value: f32,
    ) -> &mut Self {
        let elements = shape.iter().copied().product::<usize>();
        let begin = self.payload.len();

        for _ in 0..elements {
            self.payload.extend_from_slice(&value.to_le_bytes());
        }

        self.declare(name, DType::F32, shape, begin)
    }

    /// Per-element writer for oracle fixtures that pin varied source words at exact positions.
    pub(crate) fn add_with(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: &[usize],
        mut byte_at: impl FnMut(usize) -> u8,
    ) -> &mut Self {
        let elements = shape.iter().copied().product::<usize>();
        let bytes = elements * dtype.byte_width() as usize;
        let begin = self.payload.len();

        self.payload.extend((0..bytes).map(&mut byte_at));
        self.declare(name, dtype, shape, begin)
    }

    /// Header document and payload, for fixtures whose tests perturb the declaration or the bytes.
    pub(crate) fn into_parts(self) -> (Value, Vec<u8>) {
        (Value::Object(self.header), self.payload)
    }

    /// Complete safetensors file: 8-byte little-endian header length, the space-padded JSON
    /// header, then the payload. The padding matches the pinned checkpoint shards, which align
    /// every payload to eight bytes.
    pub(crate) fn finish(self) -> Vec<u8> {
        let mut header = serde_json::to_vec(&Value::Object(self.header)).unwrap();

        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }

        let mut file = Vec::with_capacity(8 + header.len() + self.payload.len());

        file.extend_from_slice(&(header.len() as u64).to_le_bytes());
        file.extend_from_slice(&header);
        file.extend_from_slice(&self.payload);
        file
    }

    /// Writes the complete file at `path`.
    pub(crate) fn write(self, path: &Path) {
        fs::write(path, self.finish()).unwrap();
    }

    fn declare(
        &mut self,
        name: impl Into<String>,
        dtype: DType,
        shape: &[usize],
        begin: usize,
    ) -> &mut Self {
        self.header.insert(
            name.into(),
            json!({
                "dtype": dtype.as_str(),
                "shape": shape,
                "data_offsets": [begin, self.payload.len()]
            }),
        );
        self
    }
}
