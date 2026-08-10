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

/// IEEE-754 binary16 from f32 (round-to-nearest-even).
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mut mant = bits & 0x7F_FFFF;

    if exp == 255 {
        return if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7C00 | ((mant >> 13) as u16)
        };
    }

    exp -= 127;
    if exp > 15 {
        return sign | 0x7C00; // overflow → inf
    }
    if exp < -14 {
        // subnormal / zero
        if exp < -24 {
            return sign;
        }
        mant |= 0x80_0000;
        let shift = (-14 - exp) as u32 + 13;
        let rounded = (mant + (1 << (shift - 1))) >> shift;
        return sign | (rounded as u16);
    }

    let half_exp = (exp + 15) as u16;
    let half_mant = mant + 0x1000; // round
    if half_mant & 0x80_0000 != 0 {
        // mantissa overflow into exp
        return sign | ((half_exp + 1) << 10);
    }
    sign | (half_exp << 10) | ((half_mant >> 13) as u16)
}
