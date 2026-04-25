use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use memmap2::Mmap;
use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum HeaderEntry {
    Tensor(TensorInfo),
    Metadata(HashMap<String, String>),
}

pub type SafetensorHeader = HashMap<String, HeaderEntry>;
pub type Weights = [f32]; // TODO: For now we support only f32 weights

#[derive(Error, Debug)]
pub enum SafetensorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Metadata not found")]
    MetadataNotFound,

    #[error("Key '{0}' is Metadata, not a Tensor. For getting Metadata calll get_metadata")]
    IsMetadata(String),

    #[error("Key '{0}' is Tensor, not a Metadata. For getting Tensor calll get_tensor")]
    IsTensor(String),

    #[error("Casting error: {0}")]
    CastError(#[from] bytemuck::PodCastError),

    #[error("Offset overflow: Header size + tensor offsets exceed memory limits")]
    OffsetOverflow,

    #[error("Unsupported dtype: expected F32, found {0}")]
    DtypeMismatch(String),
}

pub struct SafeTensor {
    header_size: u64,
    header: SafetensorHeader,
    data: Mmap,
}

impl SafeTensor {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, SafetensorError> {
        let mut file = File::open(path)?;

        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf)?;
        let header_size = u64::from_le_bytes(size_buf);

        let mut json_buffer = vec![0u8; header_size as usize];
        file.read_exact(&mut json_buffer)?;
        let json_str = String::from_utf8(json_buffer)?;

        let header: SafetensorHeader = serde_json::from_str(&json_str)?;

        let mmap = unsafe { Mmap::map(&file)? };

        Ok(SafeTensor {
            header_size,
            header,
            data: mmap,
        })
    }

    pub fn get_header(&self) -> &SafetensorHeader {
        &self.header
    }

    pub fn get_tensor(&self, key: &str) -> Result<&Weights, SafetensorError> {
        let header_entry = self
            .header
            .get(key)
            .ok_or_else(|| SafetensorError::TensorNotFound(key.to_string()))?;

        let tensor_header = match header_entry {
            HeaderEntry::Tensor(info) => info,
            HeaderEntry::Metadata(_) => return Err(SafetensorError::IsMetadata(key.to_string())),
        };

        if tensor_header.dtype != "F32" {
            return Err(SafetensorError::DtypeMismatch(tensor_header.dtype.clone()));
        }

        let data_start = self
            .header_size
            .checked_add(8)
            .ok_or(SafetensorError::OffsetOverflow)?;

        let start = data_start
            .checked_add(tensor_header.data_offsets.0 as u64)
            .ok_or(SafetensorError::OffsetOverflow)? as usize;

        let end = data_start
            .checked_add(tensor_header.data_offsets.1 as u64)
            .ok_or(SafetensorError::OffsetOverflow)? as usize;

        let tensor_bytes = self
            .data
            .get(start..end)
            .ok_or(SafetensorError::OffsetOverflow)?;

        let weights: &[f32] = bytemuck::try_cast_slice(tensor_bytes)?;

        Ok(weights)
    }

    pub fn get_metadata(&self) -> Result<&HashMap<String, String>, SafetensorError> {
        let key = "__metadata__";

        match self.header.get(key) {
            Some(HeaderEntry::Metadata(m)) => Ok(m),
            Some(HeaderEntry::Tensor(_)) => Err(SafetensorError::IsTensor(key.to_string())),
            None => Err(SafetensorError::MetadataNotFound),
        }
    }
}
