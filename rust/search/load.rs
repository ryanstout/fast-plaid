use anyhow::Result;
use once_cell::sync::Lazy;
use parking_lot::{Condvar, Mutex};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_tch::PyTensor;

use crate::search::tensor::StridedTensor;
use crate::utils::errors::anyhow_to_pyerr;
use crate::utils::residual_codec::ResidualCodec;

static INDEX_LOCKS: Lazy<Mutex<HashMap<String, Arc<IndexPathLock>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Default)]
struct IndexLockState {
    active_readers: usize,
    waiting_writers: usize,
    writer_active: bool,
}

#[derive(Default)]
struct IndexPathLock {
    state: Mutex<IndexLockState>,
    changed: Condvar,
}

enum IndexLockMode {
    Read,
    Write,
}

pub struct IndexLockGuard {
    lock: Arc<IndexPathLock>,
    mode: IndexLockMode,
}

fn lock_for_index(index_path: &str) -> Arc<IndexPathLock> {
    let mut locks = INDEX_LOCKS.lock();
    locks
        .entry(index_path.to_string())
        .or_insert_with(|| Arc::new(IndexPathLock::default()))
        .clone()
}

pub fn acquire_index_read(index_path: &str) -> IndexLockGuard {
    let lock = lock_for_index(index_path);
    {
        let mut state = lock.state.lock();
        while state.writer_active || state.waiting_writers > 0 {
            lock.changed.wait(&mut state);
        }
        state.active_readers += 1;
    }
    IndexLockGuard {
        lock,
        mode: IndexLockMode::Read,
    }
}

fn acquire_index_write(index_path: &str) -> IndexLockGuard {
    let lock = lock_for_index(index_path);
    {
        let mut state = lock.state.lock();
        state.waiting_writers += 1;
        while state.writer_active || state.active_readers > 0 {
            lock.changed.wait(&mut state);
        }
        state.waiting_writers -= 1;
        state.writer_active = true;
    }
    IndexLockGuard {
        lock,
        mode: IndexLockMode::Write,
    }
}

pub fn index_write_lock_for_rust(index_path: &str) -> IndexLockGuard {
    acquire_index_write(index_path)
}

impl Drop for IndexLockGuard {
    fn drop(&mut self) {
        let mut state = self.lock.state.lock();
        match self.mode {
            IndexLockMode::Read => {
                state.active_readers -= 1;
            }
            IndexLockMode::Write => {
                state.writer_active = false;
            }
        }
        self.lock.changed.notify_all();
    }
}

#[pyclass]
pub struct PyIndexWriteLock {
    guard: Option<IndexLockGuard>,
}

#[pymethods]
impl PyIndexWriteLock {
    fn release(&mut self) {
        self.guard.take();
    }
}

impl Drop for PyIndexWriteLock {
    fn drop(&mut self) {
        self.guard.take();
    }
}

#[pyfunction]
pub fn index_write_lock(py: Python<'_>, index_path: String) -> PyResult<PyIndexWriteLock> {
    let guard = py.allow_threads(move || acquire_index_write(&index_path));
    Ok(PyIndexWriteLock { guard: Some(guard) })
}

/// Parses a Python-style device string into a `tch::Device`.
///
/// Supports "cpu", "cuda", and specific GPU indices like "cuda:1".
pub fn get_device(device: &str) -> Result<Device, PyErr> {
    match device.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::Cuda(0)),
        s if s.starts_with("cuda:") => {
            let parts: Vec<&str> = s.split(':').collect();
            if parts.len() == 2 {
                parts[1].parse::<usize>().map(Device::Cuda).map_err(|_| {
                    PyValueError::new_err(format!("Invalid CUDA device index: '{}'", parts[1]))
                })
            } else {
                Err(PyValueError::new_err(
                    "Invalid CUDA device format. Expected 'cuda:N'.",
                ))
            }
        },
        _ => Err(PyValueError::new_err(format!(
            "Unsupported device string: '{}'",
            device
        ))),
    }
}

#[derive(Deserialize, Debug)]
pub struct Metadata {
    pub num_chunks: usize,
    pub nbits: i64,
}

/// The core struct holding all immutable data required for search operations.
///
/// This struct is designed to be shared across threads. It contains the
/// quantization codec (centroids, weights) and the document index structures
/// (IVF lists, compressed codes, and residuals).
pub struct LoadedIndex {
    pub codec: ResidualCodec,
    pub ivf_index_strided: Option<StridedTensor>,
    pub doc_codes_strided: StridedTensor,
    pub doc_residuals_strided: StridedTensor,
    pub nbits: i64,
}

unsafe impl Send for LoadedIndex {}
unsafe impl Sync for LoadedIndex {}

/// A wrapper around the Rust `LoadedIndex` struct that can be held by Python.
///
/// This wrapper allows the Python runtime to manage the lifetime of the
/// underlying Rust index structure. When the Python object is garbage collected,
/// the Rust memory is freed.
#[pyclass]
pub struct PyLoadedIndex {
    pub index_path: String,
    pub inner: LoadedIndex,
}

/// Ensures the tensor is on the target device and kind without copying if not necessary.
///
/// This is critical for memory-mapped tensors. Calling `to_device` blindly on a
/// CPU mmap tensor will force a load into RAM, even if the target is also CPU.
fn ensure_tensor(t: PyTensor, device: Device, kind: Kind) -> Tensor {
    // PyTensor derefs to Tensor. We take a shallow reference first.
    let mut res: Tensor = t.shallow_clone();

    // Only convert device if different (avoids copy/move overhead)
    if res.device() != device {
        res = res.to_device(device);
    }

    // Only convert kind if different (avoids casting overhead)
    if res.kind() != kind {
        res = res.to_kind(kind);
    }

    res
}

/// Constructs the internal Index object from raw tensors loaded in Python.
///
/// This function acts as the bridge between Python's file loading and Rust's
/// search engine. It organizes the raw tensors into a `LoadedIndex` struct.
///
///
///
/// # Key Behavior
/// - **Zero-Copy Optimization**: If `device` is "cpu", large tensors (codes, residuals)
///   are assumed to be memory-mapped. The function verifies padding and uses them
///   directly without allocation.
/// - **Codec Handling**: Small tensors (centroids, weights) are loaded into RAM/VRAM
///   immediately for fast lookup during decompression.
/// - **Low Memory Mode**: If `low_memory` is true, the large document tensors are strictly
///   kept on the CPU, even if the target `device` is CUDA.
///
/// # Arguments
///
/// * `nbits` - The quantization bit-width (e.g., 2 or 4).
/// * `centroids` - The coarse centroids (float16).
/// * `avg_residual` - The average residual vector (float16).
/// * `bucket_cutoffs` - Quantization bucket boundaries (float16).
/// * `bucket_weights` - Quantization bucket values (float16).
/// * `ivf` - The Inverted File index structure (int64). None for compress-only indices.
/// * `ivf_lengths` - Lengths of the IVF lists (int32). None for compress-only indices.
/// * `doc_codes` - The compressed document codes (int64).
/// * `doc_residuals` - The compressed document residuals (uint8).
/// * `doc_lengths` - The true lengths of documents (int64).
/// * `device` - The target device string (e.g. "cuda:0").
/// * `low_memory` - If true, keeps document data on CPU.
#[pyfunction]
#[pyo3(signature = (
    nbits,
    centroids,
    avg_residual,
    bucket_cutoffs,
    bucket_weights,
    ivf,
    ivf_lengths,
    doc_codes,
    doc_residuals,
    doc_lengths,
    device,
    low_memory,
    index_path = None
))]
#[allow(clippy::too_many_arguments)]
pub fn construct_index(
    py: Python<'_>,
    nbits: i64,
    centroids: PyTensor,
    avg_residual: PyTensor,
    bucket_cutoffs: PyTensor,
    bucket_weights: PyTensor,
    ivf: Option<PyTensor>,
    ivf_lengths: Option<PyTensor>,
    doc_codes: PyTensor,
    doc_residuals: PyTensor,
    doc_lengths: PyTensor,
    device: String,
    low_memory: bool,
    index_path: Option<String>,
) -> PyResult<PyLoadedIndex> {
    let main_device = get_device(&device)?;

    let loaded_index = py
        .allow_threads(move || {
            // Force document tensors to CPU in low memory mode
            let storage_device = if low_memory { Device::Cpu } else { main_device };

            // Load codec (small tensors)
            let codec = ResidualCodec::load(
                nbits,
                ensure_tensor(centroids, main_device, Kind::Half),
                ensure_tensor(avg_residual, main_device, Kind::Half),
                Some(ensure_tensor(bucket_cutoffs, main_device, Kind::Half)),
                Some(ensure_tensor(bucket_weights, main_device, Kind::Half)),
                main_device,
            )?;

            // Build IVF index (None for compress-only indices)
            let ivf_index_strided = match (ivf, ivf_lengths) {
                (Some(ivf_t), Some(ivf_len_t)) => Some(StridedTensor::new(
                    ensure_tensor(ivf_t, main_device, Kind::Int64),
                    ensure_tensor(ivf_len_t, main_device, Kind::Int),
                    main_device,
                )),
                _ => None,
            };

            // Load document data (large tensors, may stay on CPU in low memory mode)
            let doc_lens_t = ensure_tensor(doc_lengths, storage_device, Kind::Int64);
            let doc_codes_t = ensure_tensor(doc_codes, storage_device, Kind::Int64);
            let doc_residuals_t = ensure_tensor(doc_residuals, storage_device, Kind::Uint8);

            let doc_codes_strided =
                StridedTensor::new(doc_codes_t, doc_lens_t.shallow_clone(), storage_device);

            let doc_residuals_strided =
                StridedTensor::new(doc_residuals_t, doc_lens_t, storage_device);

            Ok(LoadedIndex {
                codec,
                ivf_index_strided,
                doc_codes_strided,
                doc_residuals_strided,
                nbits,
            })
        })
        .map_err(anyhow_to_pyerr)?;

    Ok(PyLoadedIndex {
        index_path: index_path.unwrap_or_default(),
        inner: loaded_index,
    })
}
