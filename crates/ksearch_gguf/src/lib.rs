//! GGUF mmap + CPU dequant (ported from metal-llm-server boilerplate).

mod gguf_impl;
mod tokenizer;

pub use gguf_impl::*;
pub use tokenizer::*;

#[inline]
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(value: f32) -> u16 {
    ((value.to_bits() + 0x8000) >> 16) as u16
}

#[inline]
pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value >> 15) as f32) * -2.0 + 1.0;
    let exp = (value >> 10) & 0x1F;
    let mant = value & 0x3FF;
    if exp == 0 {
        sign * (mant as f32) * (2.0f32).powi(-24)
    } else if exp == 31 {
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + (mant as f32) / 1024.0) * (2.0f32).powi(exp as i32 - 15)
    }
}
