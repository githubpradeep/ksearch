//! Tinygrad-shaped IR: Graph primitives + Kernel IR. No fused-kernel catalog.

mod graph;
mod kernel;

pub use graph::*;
pub use kernel::*;

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    /// GGML Q4_K packed (256 elems / 144 bytes). Logical shape is element counts.
    Q4K,
    Q5K,
    Q6K,
    Q40,
    BF16,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::Q4K | DType::Q5K | DType::Q6K | DType::Q40 => 0,
        }
    }

    pub fn msl(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::BF16 => "bfloat",
            DType::Q4K | DType::Q5K | DType::Q6K | DType::Q40 => "uchar",
        }
    }
}

pub fn q40_row_bytes(hd: usize) -> usize {
    assert!(hd % 32 == 0, "Q4_0 hd must be multiple of 32");
    (hd / 32) * 18
}

pub fn q40_nbytes(max_t: usize, hd: usize) -> usize {
    max_t * q40_row_bytes(hd)
}

pub fn q4k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q4_K nelem must be multiple of 256");
    (nelem / 256) * 144
}

pub fn q5k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q5_K nelem must be multiple of 256");
    (nelem / 256) * 176
}

pub fn q6k_nbytes(nelem: usize) -> usize {
    assert!(nelem % 256 == 0, "Q6_K nelem must be multiple of 256");
    (nelem / 256) * 210
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IrError {
    #[error("bad tensor id")]
    BadTensorId,
    #[error("shape / dtype mismatch")]
    ShapeMismatch,
    #[error("bad reduce axis")]
    BadAxis,
}

impl fmt::Display for TensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{}", self.0)
    }
}
