//! Build HuggingFace `tokenizers::Tokenizer` from GGUF embedded BPE metadata.

use crate::{token_type, Gguf};
use ahash::AHashMap;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Tokenizer};

/// Build a tokenizer from `tokenizer.ggml.*` keys in a GGUF (Gemma4 / Gemma / Llama).
pub fn build_tokenizer_from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
    let model = g.get_str("tokenizer.ggml.model").unwrap_or("");
    if !matches!(model, "gemma4" | "llama" | "gemma" | "gemma3") {
        return Err(format!("unexpected tokenizer.ggml.model '{model}'"));
    }

    let tokens = g
        .get_arr_str("tokenizer.ggml.tokens")
        .ok_or("tokenizer.ggml.tokens missing")?;
    let merges = g
        .get_arr_str("tokenizer.ggml.merges")
        .ok_or("tokenizer.ggml.merges missing")?;
    let token_types = g.get_arr_i32("tokenizer.ggml.token_type");

    let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }

    let merges_vec: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();

    let unk_token = g
        .get_u32("tokenizer.ggml.unknown_token_id")
        .and_then(|id| tokens.get(id as usize).cloned())
        .unwrap_or_else(|| "<unk>".to_string());

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges_vec)
        .unk_token(unk_token)
        .byte_fallback(true)
        .fuse_unk(true)
        .build()
        .map_err(|e| format!("BPE build failed: {e}"))?;

    let mut tok = Tokenizer::new(bpe);

    let prepend = if g
        .get_bool("tokenizer.ggml.add_space_prefix")
        .unwrap_or(false)
    {
        PrependScheme::First
    } else {
        PrependScheme::Never
    };
    tok.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', prepend, true)));
    tok.with_decoder(Some(Metaspace::new('\u{2581}', prepend, true)));

    if let Some(types) = token_types {
        let specials: Vec<AddedToken> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    types.get(*i).copied(),
                    Some(token_type::CONTROL) | Some(token_type::USER_DEFINED)
                )
            })
            .map(|(_, t)| AddedToken::from(t.clone(), true))
            .collect();
        tok.add_special_tokens(&specials);
    }

    if g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(true) {
        if let Some(bos_id) = g.get_u32("tokenizer.ggml.bos_token_id") {
            if let Some(bos_tok) = tokens.get(bos_id as usize) {
                let post = TemplateProcessing::builder()
                    .try_single(format!("{} $A", bos_tok))
                    .map_err(|e| format!("BOS template: {e}"))?
                    .special_tokens(vec![(bos_tok.as_str(), bos_id)])
                    .build()
                    .map_err(|e| format!("BOS processor: {e}"))?;
                tok.with_post_processor(Some(post));
            }
        }
    }

    Ok(tok)
}

/// Gemma4 E2B/E4B chat template (thinking off): user turn + open model turn.
pub fn gemma4_chat_prompt(user: &str) -> String {
    format!("<|turn>user\n{user}<turn|>\n<|turn>model\n")
}

/// Multi-turn Gemma4 chat template. `messages` are `(role, content)` with roles
/// `system` / `user` / `assistant`. Always opens a `model` generation turn.
pub fn gemma4_chat_from_messages<R: AsRef<str>, C: AsRef<str>>(messages: &[(R, C)]) -> String {
    let mut system = String::new();
    let mut body = String::new();
    for (role, content) in messages {
        let role = role.as_ref();
        let content = content.as_ref();
        if role == "system" {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(content);
            continue;
        }
        let mapped = if role == "assistant" { "model" } else { "user" };
        body.push_str("<|turn>");
        body.push_str(mapped);
        body.push('\n');
        body.push_str(content);
        body.push_str("<turn|>\n");
    }
    let mut prompt = String::new();
    if !system.is_empty() {
        prompt.push_str("<|turn>system\n");
        prompt.push_str(&system);
        prompt.push_str("<turn|>\n");
    }
    prompt.push_str(&body);
    prompt.push_str("<|turn>model\n");
    prompt
}

pub fn encode_prompt(tok: &Tokenizer, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
    let enc = tok
        .encode(text, add_special)
        .map_err(|e| format!("encode failed: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_chat_from_messages_opens_model_turn() {
        let prompt = gemma4_chat_from_messages(&[
            ("system", "be brief"),
            ("user", "Hi"),
            ("assistant", "Hi!"),
            ("user", "again"),
        ]);
        assert_eq!(
            prompt,
            "<|turn>system\nbe brief<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHi!<turn|>\n<|turn>user\nagain<turn|>\n<|turn>model\n"
        );
        assert_eq!(
            gemma4_chat_prompt("Hi"),
            "<|turn>user\nHi<turn|>\n<|turn>model\n"
        );
    }
}
