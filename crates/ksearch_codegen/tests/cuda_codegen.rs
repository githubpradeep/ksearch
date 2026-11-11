//! CUDA backend smoke + syntax validation.
//!
//! The CUDA backend reuses the MSL-flavoured emitter and rewrites the few
//! constructs CUDA has no equivalent for. These tests take the shared kernel zoo
//! ([`ksearch_codegen::smoke::smoke_kernels`], one `KernelIr` per launch kind /
//! helper family), render it for both targets, and check the structural
//! properties of the CUDA output (no Metal attributes, shared memory declared as
//! `__shared__`, argument order, ...).
//!
//! When `clang++` is available the generated CUDA source is additionally fed
//! through a host C++ parse (`-x c++ -fsyntax-only`). The shim in
//! `ksearch_codegen::cuda_prelude` defines host stubs for the CUDA builtins, so
//! this exercises the full translation unit — prelude, helpers and kernel — and
//! catches dialect leaks without needing a GPU or `nvcc`.
//!
//! Set `KSEARCH_SKIP_CUDA_SYNTAX=1` to skip the `clang++` step.
//! `KSEARCH_DUMP_DIR=/tmp/out` writes every rendered source for `nvcc`/`metal`.

use ksearch_codegen::{render_cuda, render_for, render_msl, Target};
use ksearch_ir::{KernelIr, KirStmt, OptSchedule};
use std::io::Write;
use std::process::Command;

/// Every kernel family that reaches the CUDA backend.
fn cases() -> Vec<KernelIr> {
    ksearch_codegen::smoke::smoke_kernels()
}

/// The parameter list of the `KS_KERNEL void <name>(...)` entry point.
fn cuda_signature(source: &str) -> Vec<String> {
    let start = source
        .find("KS_KERNEL void")
        .expect("kernel entry point missing");
    let open = source[start..].find('(').expect("signature `(` missing") + start;
    let close = source[open..].find(')').expect("signature `)` missing") + open;
    source[open + 1..close]
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[test]
fn cuda_output_is_not_metal() {
    for k in cases() {
        let cuda = render_cuda(&k, OptSchedule::default()).expect("render cuda");
        let metal = render_msl(&k, OptSchedule::default()).expect("render metal");

        assert!(
            cuda.source.contains("ksearch CUDA dialect shim"),
            "{}: CUDA prelude missing",
            k.name
        );
        assert!(
            !cuda.source.contains("[[buffer("),
            "{}: Metal buffer attributes leaked into CUDA",
            k.name
        );
        assert!(
            !cuda.source.contains("[[thread_position_in_grid]]"),
            "{}: Metal index attributes leaked into CUDA",
            k.name
        );
        assert!(
            !cuda.source.contains("#include <metal_stdlib>"),
            "{}: Metal header leaked into CUDA",
            k.name
        );
        assert!(
            cuda.source.contains("KS_KERNEL void"),
            "{}: kernel entry point not CUDA-qualified",
            k.name
        );
        assert!(
            !cuda.source.contains("#define device"),
            "{}: a macro named `device` makes nvcc erase __device__",
            k.name
        );
        assert!(
            cuda.source.contains("#define KS_DEVICE __device__\n"),
            "{}: device annotation macro missing",
            k.name
        );
        assert!(
            !cuda.source.contains("#define KS_DEVICE __device__ __forceinline__"),
            "{}: nvcc rejects __forceinline__ inline as a duplicate specifier",
            k.name
        );
        // Metal keeps using the MSL spellings.
        assert!(
            metal.source.contains("#include <metal_stdlib>"),
            "{}: Metal source lost its header",
            k.name
        );
        assert!(
            metal.source.contains("kernel void"),
            "{}: Metal entry point changed",
            k.name
        );
        // Shared-memory declarations go through the `KS_SMEM` indirection.
        if k.body
            .iter()
            .any(|s| matches!(s, KirStmt::TgDeclF32 { .. }))
        {
            assert!(
                cuda.source.contains("KS_SMEM float tg0[")
                    && cuda.source.contains("#define KS_SMEM __shared__"),
                "{}: CUDA shared memory not declared __shared__",
                k.name
            );
            assert!(
                metal.source.contains("KS_SMEM float tg0[")
                    && metal.source.contains("#define KS_SMEM threadgroup"),
                "{}: Metal shared memory not declared threadgroup",
                k.name
            );
        }
    }
}

/// The runtime marshals arguments positionally as `in0..inN-1` then the outputs;
/// if codegen ever reorders or drops a parameter the launch would silently read
/// garbage, so pin the ABI here.
#[test]
fn cuda_signature_matches_buffer_counts() {
    for k in cases() {
        let cuda = render_cuda(&k, OptSchedule::default()).expect("render cuda");
        let params = cuda_signature(&cuda.source);
        assert_eq!(
            params.len(),
            k.n_inputs + k.n_outputs,
            "{}: signature {params:?} does not match {} inputs + {} outputs",
            k.name,
            k.n_inputs,
            k.n_outputs
        );
        for (i, p) in params.iter().take(k.n_inputs).enumerate() {
            assert!(
                p.contains(&format!("in{i}")),
                "{}: input {i} is not `in{i}` (got `{p}`)",
                k.name
            );
            assert!(
                p.starts_with("const"),
                "{}: input {i} is not read-only (got `{p}`)",
                k.name
            );
        }
        for (j, p) in params.iter().skip(k.n_inputs).enumerate() {
            assert!(
                p.contains("out"),
                "{}: output {j} is not named `out` (got `{p}`)",
                k.name
            );
            assert!(
                !p.starts_with("const"),
                "{}: output {j} is const (got `{p}`)",
                k.name
            );
        }
    }
}

/// Every buffer the kernel reads must be a parameter, and the generated source
/// must not contain any `[[buffer(n)]]` index that CUDA would have to ignore.
#[test]
fn cuda_has_no_argument_buffer_indices() {
    for k in cases() {
        let cuda = render_cuda(&k, OptSchedule::default()).expect("render cuda");
        assert!(
            !cuda.source.contains("[[buffer"),
            "{}: argument buffer index left in CUDA source",
            k.name
        );
        assert!(
            !cuda.source.contains("[[threadgroup"),
            "{}: Metal threadgroup attribute left in CUDA source",
            k.name
        );
    }
}

#[test]
fn cuda_mul_mm_uses_cuda_helper() {
    let cuda = render_cuda(&cases()[3], OptSchedule::default()).expect("render cuda");
    assert!(
        cuda.source.contains("ksearch_q4k_mul_mm("),
        "CUDA must call the CUDA mul_mm helper"
    );
    // The Metal-only simdgroup MMA must not be pulled in.
    assert!(
        !cuda.source.contains("simdgroup_multiply_accumulate"),
        "CUDA pulled in the Metal simdgroup MMA helper"
    );
    assert!(
        !cuda.source.contains("ksearch_q4k_dequant16"),
        "CUDA pulled in the Metal MMA staging helper"
    );
    assert!(
        cuda.source.contains("mma_sync"),
        "CUDA mul_mm must multiply on tensor cores"
    );

    let metal = render_msl(&cases()[3], OptSchedule::default()).expect("render metal");
    assert!(
        metal.source.contains("simdgroup_multiply_accumulate"),
        "Metal must keep the simdgroup MMA helper"
    );
}

#[test]
fn cuda_mul_mm_launch_is_rederived() {
    // Metal launches `(32, 4)` threadgroups over ceil(N/32) x ceil(M/64) tiles;
    // CUDA re-derives grid/block/smem from the statement's (rows, batch).
    let metal = render_msl(&cases()[3], OptSchedule::default()).expect("render metal");
    let cuda = render_cuda(&cases()[3], OptSchedule::default()).expect("render cuda");

    match metal.launch {
        ksearch_codegen::LaunchHint::MulMm { tw, nsg, .. } => {
            assert_eq!((tw, nsg), (256, 1));
        }
        other => panic!("unexpected launch {other:?}"),
    }
    match cuda.launch {
        ksearch_codegen::LaunchHint::MulMm {
            tg_x,
            tg_y,
            tw,
            nsg,
            smem,
        } => {
            // rows = 64, batch = 64 -> one 64×64 block, padded stride.
            assert_eq!((tg_x, tg_y), (1, 1));
            assert_eq!((tw, nsg), (256, 1));
            assert_eq!(smem, 36864);
        }
        other => panic!("unexpected launch {other:?}"),
    }
}

#[test]
fn dump_sources_when_requested() {
    let Some(dir) = std::env::var_os("KSEARCH_DUMP_DIR") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("mk dump dir");
    for k in cases() {
        for (ext, src) in [
            (
                "cu",
                render_cuda(&k, OptSchedule::default())
                    .expect("cuda")
                    .source,
            ),
            (
                "metal",
                render_for(&k, OptSchedule::default(), Target::Metal)
                    .expect("metal")
                    .source,
            ),
        ] {
            let path = dir.join(format!("{}.{}", k.name, ext));
            std::fs::write(&path, src).expect("write");
            eprintln!("wrote {}", path.display());
        }
    }
}

#[test]
fn cuda_is_deterministic() {
    for k in cases() {
        let a = render_cuda(&k, OptSchedule::default()).expect("render a");
        let b = render_for(&k, OptSchedule::default(), Target::Cuda).expect("render b");
        assert_eq!(
            a.source, b.source,
            "{}: CUDA output not deterministic",
            k.name
        );
    }
}

/// Host-C++ syntax check of the generated CUDA for every case.
#[test]
fn cuda_source_parses_as_cxx() {
    if std::env::var("KSEARCH_SKIP_CUDA_SYNTAX").is_ok() {
        eprintln!("skipped: KSEARCH_SKIP_CUDA_SYNTAX set");
        return;
    }
    let clang = ["clang++", "c++"].into_iter().find(|c| {
        Command::new(c)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    });
    let Some(clang) = clang else {
        eprintln!("skipped: no C++ compiler found");
        return;
    };

    let dir = std::env::temp_dir().join(format!("ksearch_cuda_check_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mk temp dir");

    for k in cases() {
        let src = render_cuda(&k, OptSchedule::default())
            .expect("render cuda")
            .source;
        let path = dir.join(format!("{}.cu.cpp", k.name));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(src.as_bytes()))
            .expect("write source");

        let out = Command::new(clang)
            .args(["-std=c++17", "-x", "c++", "-fsyntax-only"])
            .arg("-Wno-unused-variable")
            .arg("-Wno-unused-parameter")
            .arg("-Wno-unused-but-set-variable")
            .arg("-Wno-unknown-pragmas")
            .arg(&path)
            .output()
            .expect("run compiler");
        assert!(
            out.status.success(),
            "{}: generated CUDA failed the host syntax check:\n{}\n--- source ---\n{}",
            k.name,
            String::from_utf8_lossy(&out.stderr),
            src
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
