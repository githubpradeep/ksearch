//! `LaunchHint` → CUDA grid/block/shared-memory mapping.
//!
//! Mirrors `ksearch_metal`'s dispatch policy so a kernel tuned for one backend
//! launches the same work on the other:
//!
//! | `LaunchHint`        | grid                          | block          |
//! |---------------------|-------------------------------|----------------|
//! | `Elementwise{n}`    | `ceil(n / tg)`                | `tg`           |
//! | `Rows{rows}`        | `ceil(rows / tg)`             | `tg`           |
//! | `RowsParallel`      | `rows`                        | `tg`           |
//! | `RowsParallelSg`    | `rows`                        | `(32, nsg)`    |
//! | `RowsParallel2D`    | `(rows, batch)`               | `tg`           |
//! | `MulMm`             | `(tg_x, tg_y)`                | `(tw, nsg)`    |
//!
//! `RowsParallelSg` and `MulMm` keep the 2D block shape because the generated
//! bodies derive `tiisg`/`sgitg` from `threadIdx.x`/`threadIdx.y`.

use ksearch_codegen::LaunchHint;

/// Everything `cuLaunchKernel` needs, in driver-ready form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    /// Dynamic shared memory in bytes.
    pub smem: u32,
}

impl LaunchConfig {
    pub fn new(grid: (u32, u32), block: (u32, u32), smem: u32) -> Self {
        Self {
            grid: (grid.0, grid.1, 1),
            block: (block.0, block.1, 1),
            smem,
        }
    }

    /// Total threads per block.
    pub fn threads(&self) -> u64 {
        self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64
    }

    /// Total blocks.
    pub fn blocks(&self) -> u64 {
        self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64
    }
}

fn ceil_div(a: u64, b: u64) -> u32 {
    if b == 0 {
        return 1;
    }
    a.div_ceil(b).max(1) as u32
}

/// Map a [`LaunchHint`] to a CUDA launch.
///
/// `tg_size` is the default block size for flat kernels (metal's
/// `threadgroup` default, typically 64/128/256).
pub fn launch_config(hint: &LaunchHint, tg_size: u64) -> LaunchConfig {
    let tg = tg_size.max(1);
    match hint {
        LaunchHint::Elementwise { n } => {
            let t = tg.min(*n as u64).max(1);
            LaunchConfig::new((ceil_div(*n as u64, t), 1), (t as u32, 1), 0)
        }
        LaunchHint::Rows { rows, .. } => {
            let t = tg.min(*rows as u64).max(1);
            LaunchConfig::new((ceil_div(*rows as u64, t), 1), (t as u32, 1), 0)
        }
        LaunchHint::RowsParallel { rows, tg } => {
            LaunchConfig::new(((*rows).max(1) as u32, 1), (u32c(*tg), 1), 0)
        }
        LaunchHint::RowsParallelSg { rows, nsg } => {
            LaunchConfig::new(((*rows).max(1) as u32, 1), (32, u32c(*nsg)), 0)
        }
        LaunchHint::RowsParallel2D { rows, batch, tg } => LaunchConfig::new(
            ((*rows).max(1) as u32, (*batch).max(1) as u32),
            (u32c(*tg), 1),
            0,
        ),
        LaunchHint::MulMm {
            tg_x,
            tg_y,
            tw,
            nsg,
            smem,
        } => LaunchConfig::new(
            (u32c(*tg_x), u32c(*tg_y)),
            (u32c(*tw), u32c(*nsg)),
            (*smem).min(u32::MAX as u64) as u32,
        ),
    }
}

/// Saturating `u64` → `u32` for launch dimensions.
fn u32c(v: u64) -> u32 {
    v.max(1).min(u32::MAX as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_covers_all_elements() {
        let c = launch_config(&LaunchHint::Elementwise { n: 1000 }, 256);
        assert_eq!(c.block, (256, 1, 1));
        assert_eq!(c.grid, (4, 1, 1));
        // 4 * 256 >= 1000
        assert!(c.blocks() * c.threads() >= 1000);
    }

    #[test]
    fn elementwise_shrinks_block_for_tiny_grids() {
        let c = launch_config(&LaunchHint::Elementwise { n: 3 }, 256);
        assert_eq!(c.block, (3, 1, 1));
        assert_eq!(c.grid, (1, 1, 1));
    }

    #[test]
    fn rows_parallel_sg_is_32_by_nsg() {
        let c = launch_config(&LaunchHint::RowsParallelSg { rows: 64, nsg: 4 }, 128);
        assert_eq!(c.block, (32, 4, 1));
        assert_eq!(c.grid, (64, 1, 1));
        assert_eq!(c.threads(), 128);
    }

    #[test]
    fn mul_mm_keeps_tile_and_smem() {
        let c = launch_config(
            &LaunchHint::MulMm {
                tg_x: 3,
                tg_y: 2,
                tw: 256,
                nsg: 1,
                smem: 8192,
            },
            128,
        );
        assert_eq!(c.grid, (3, 2, 1));
        assert_eq!(c.block, (256, 1, 1));
        assert_eq!(c.smem, 8192);
    }

    #[test]
    fn rows_parallel_2d_expands_batch_on_y() {
        let c = launch_config(
            &LaunchHint::RowsParallel2D {
                rows: 16,
                batch: 5,
                tg: 64,
            },
            128,
        );
        assert_eq!(c.grid, (16, 5, 1));
        assert_eq!(c.block, (64, 1, 1));
    }

    #[test]
    fn zero_sizes_never_produce_empty_launches() {
        for hint in [
            LaunchHint::Elementwise { n: 0 },
            LaunchHint::Rows { rows: 0, cols: 1 },
            LaunchHint::RowsParallel { rows: 0, tg: 0 },
            LaunchHint::RowsParallelSg { rows: 0, nsg: 0 },
        ] {
            let c = launch_config(&hint, 0);
            assert!(c.blocks() >= 1 && c.threads() >= 1, "{hint:?} -> {c:?}");
        }
    }
}
