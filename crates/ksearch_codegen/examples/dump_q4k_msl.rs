//! Dump generated MSL for Q4K gate_up to inspect coop expand.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ksearch_codegen::lower_to_metal_chip;
    use ksearch_ir::{DType, Graph, Shape};

    let mut g = Graph::new();
    let wg = g.input(Shape(vec![6144, 1536]), DType::Q4K);
    let wu = g.input(Shape(vec![6144, 1536]), DType::Q4K);
    let x = g.input(Shape(vec![1536]), DType::F16);
    let out = g.matvec_gate_up_gelu(wg, wu, x)?;
    let k = lower_to_metal_chip(&g, out, "dump")?;
    println!("launch={:?} bytes={}", k.launch, k.source.len());
    if let Some(idx) = k.source.find("kernel void") {
        print!("{}", &k.source[idx..]);
    }
    Ok(())
}
