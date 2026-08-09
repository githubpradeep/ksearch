use ksearch_gguf::{Gguf, Vocab};
fn main() {
    let path = std::env::args().nth(1).expect("gguf path");
    let g = Gguf::open(&path);
    let v = Vocab::from_gguf(&g).expect("vocab");
    for i in 100..120 {
        println!("{i}: {:?}", v.piece(i as u32));
    }
    // find Hi / Hello / user / model pieces
    for want in ["user", "model", "Hi", "Hello", "▁Hi", "▁Hello", "▁user", "▁model"] {
        for i in 0..v.len() {
            if v.piece(i as u32) == Some(want) {
                println!("FOUND {want} -> {i}");
            }
        }
    }
}
