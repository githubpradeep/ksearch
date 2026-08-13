//! Gemma 4 chat template + Pi/OpenAI tool-calling (parity with metal-llm-server).

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TURN_START: &str = "<|turn>";
pub const TURN_END: &str = "<turn|>";
pub const CHANNEL_START: &str = "<|channel>";
pub const CHANNEL_END: &str = "<channel|>";
const GEMMA4_TOOL_START: &str = "<|tool>";
const GEMMA4_TOOL_END: &str = "<tool|>";
pub const NATIVE_TOOL_CALL_PREFIX: &str = "<|tool_call>call:";
pub const NATIVE_TOOL_CALL_TRIGGER: &str = "<|tool_call>";
pub const NATIVE_TOOL_CALL_SUFFIX: &str = "<tool_call|>";
const NATIVE_TOOL_RESPONSE_PREFIX: &str = "<|tool_response>response:";
const NATIVE_TOOL_RESPONSE_SUFFIX: &str = "<tool_response|>";

#[derive(Debug, Deserialize, Clone)]
pub struct Tool {
    #[serde(rename = "type", default)]
    #[allow(dead_code)]
    pub tool_type: Option<String>,
    pub function: FunctionDef,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_content")]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_call_id: Option<String>,
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ContentField {
        Text(String),
        Parts(Vec<ContentPart>),
    }
    #[derive(Deserialize)]
    struct ContentPart {
        #[serde(rename = "type", default)]
        kind: Option<String>,
        #[serde(default)]
        text: Option<String>,
    }
    Ok(match Option::<ContentField>::deserialize(deserializer)? {
        None => None,
        Some(ContentField::Text(s)) => Some(s),
        Some(ContentField::Parts(parts)) => {
            let mut out = String::new();
            for part in parts {
                if part.kind.as_deref().unwrap_or("text") == "text" {
                    if let Some(text) = part.text {
                        out.push_str(&text);
                    }
                }
            }
            Some(out)
        }
    })
}

fn gemma4_string(value: &str) -> String {
    format!("<|\"|>{}<|\"|>", value)
}

fn json_value_to_gemma4(value: &Value) -> String {
    match value {
        Value::String(s) => gemma4_string(s),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(items) => {
            let inner = items.iter().map(json_value_to_gemma4).collect::<Vec<_>>().join(",");
            format!("[{inner}]")
        }
        Value::Object(map) => {
            let inner = map
                .iter()
                .map(|(k, v)| format!("{k}:{}", json_value_to_gemma4(v)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

fn json_schema_to_gemma4_params(schema: &Value) -> String {
    let Some(obj) = schema.as_object() else {
        return json_value_to_gemma4(schema);
    };
    let mut members = Vec::new();
    if let Some(properties) = obj.get("properties").and_then(|v| v.as_object()) {
        let props: Vec<String> = properties
            .iter()
            .map(|(key, value)| {
                let mut parts = Vec::new();
                if let Some(description) = value.get("description").and_then(|v| v.as_str()) {
                    parts.push(format!("description:{}", gemma4_string(description)));
                }
                if let Some(typ) = value.get("type").and_then(|v| v.as_str()) {
                    parts.push(format!("type:{}", gemma4_string(&typ.to_uppercase())));
                }
                format!("{key}:{{{}}}", parts.join(","))
            })
            .collect();
        members.push(format!("properties:{{{}}}", props.join(",")));
    }
    if let Some(required) = obj.get("required").and_then(|v| v.as_array()) {
        let req: Vec<String> = required
            .iter()
            .filter_map(|value| value.as_str().map(gemma4_string))
            .collect();
        members.push(format!("required:[{}]", req.join(",")));
    }
    if let Some(typ) = obj.get("type").and_then(|v| v.as_str()) {
        members.push(format!("type:{}", gemma4_string(&typ.to_uppercase())));
    }
    if members.is_empty() {
        "{}".to_string()
    } else {
        format!("{{{}}}", members.join(","))
    }
}

fn render_tool_declarations(tools: &[Tool]) -> String {
    let mut s = String::new();
    for tool in tools {
        let f = &tool.function;
        let description = gemma4_string(f.description.as_deref().unwrap_or(""));
        let parameters = f
            .parameters
            .as_ref()
            .map(json_schema_to_gemma4_params)
            .unwrap_or_else(|| "{}".to_string());
        s.push_str(GEMMA4_TOOL_START);
        s.push_str("declaration:");
        s.push_str(&f.name);
        s.push_str("{description:");
        s.push_str(&description);
        s.push_str(",parameters:");
        s.push_str(&parameters);
        s.push('}');
        s.push_str(GEMMA4_TOOL_END);
    }
    s
}

fn render_assistant_tool_call(tc: &ToolCall) -> String {
    let args = serde_json::from_str::<Value>(&tc.function.arguments).unwrap_or_else(|_| serde_json::json!({}));
    format!(
        "{NATIVE_TOOL_CALL_PREFIX}{}{}{NATIVE_TOOL_CALL_SUFFIX}",
        tc.function.name,
        json_value_to_gemma4(&args)
    )
}

fn render_tool_response(name: &str, content: &str) -> String {
    format!(
        "{NATIVE_TOOL_RESPONSE_PREFIX}{name}{{value:{}}}{NATIVE_TOOL_RESPONSE_SUFFIX}",
        gemma4_string(content)
    )
}

pub fn tool_choice_is_required(tool_choice: Option<&Value>) -> bool {
    matches!(tool_choice, Some(Value::String(v)) if v == "required")
}

pub fn tool_choice_is_none(tool_choice: Option<&Value>) -> bool {
    matches!(tool_choice, Some(Value::String(v)) if v == "none")
}

pub fn awaits_tool_call(messages: &[ChatMessage]) -> bool {
    let mut seen_tool = false;
    for msg in messages.iter().rev() {
        match msg.role.as_str() {
            "tool" => seen_tool = true,
            "user" => {
                if msg.content.as_ref().is_some_and(|c| c.trim().is_empty()) {
                    continue;
                }
                return !seen_tool;
            }
            "assistant" if msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) => {
                return !seen_tool;
            }
            _ => {}
        }
    }
    false
}

fn should_require_tool_call(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> bool {
    if tool_choice_is_none(tool_choice) {
        return false;
    }
    if tool_choice_is_required(tool_choice) {
        return true;
    }
    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    if !has_tools {
        return false;
    }
    matches!(messages.last().map(|m| m.role.as_str()), Some("user"))
}

fn should_force_tool_call(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> bool {
    should_require_tool_call(messages, tools, tool_choice) && awaits_tool_call(messages)
}

fn last_user_content(messages: &[ChatMessage]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_deref())
}

/// Pi-style file reference: `@"path/with spaces.md"` or `@AGENTS.md`.
pub fn extract_at_file_path(content: &str) -> Option<String> {
    if let Some(start) = content.find("@\"") {
        let rest = &content[start + 2..];
        let end = rest.find('"')?;
        let path = rest[..end].trim();
        if !path.is_empty() {
            return Some(path.to_string());
        }
    }
    let at = content.find('@')?;
    let rest = &content[at + 1..];
    if rest.starts_with('"') {
        return None;
    }
    let end = rest
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    let path = rest[..end].trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':'));
    if path.is_empty() {
        return None;
    }
    Some(path.to_string())
}

fn user_wants_directory_listing(messages: &[ChatMessage]) -> bool {
    last_user_content(messages)
        .map(|content| {
            let lower = content.to_ascii_lowercase();
            lower.contains("list file")
                || lower.contains("list files")
                || lower.contains("list dir")
                || lower.contains("list directory")
                || lower.trim() == "ls"
        })
        .unwrap_or(false)
}

fn user_message_implies_read(messages: &[ChatMessage]) -> bool {
    last_user_content(messages).is_some_and(|content| {
        if extract_at_file_path(content).is_some() {
            return true;
        }
        let lower = content.to_ascii_lowercase();
        lower.contains("summar") || lower.contains("read file") || lower.contains("read the file")
    })
}

fn make_tool_call(name: &str, arguments: Value) -> ToolCall {
    ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

fn infer_tool_call_from_context(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
) -> Option<ToolCall> {
    let names: Vec<String> = tools?.iter().map(|t| t.function.name.clone()).collect();
    if names.is_empty() {
        return None;
    }
    if names.iter().any(|n| n == "bash") && user_wants_directory_listing(messages) {
        return Some(make_tool_call("bash", serde_json::json!({ "command": "ls -F" })));
    }
    if names.iter().any(|n| n == "read") && user_message_implies_read(messages) {
        if let Some(path) = last_user_content(messages).and_then(extract_at_file_path) {
            return Some(make_tool_call("read", serde_json::json!({ "path": path })));
        }
    }
    None
}

/// Tool calls we can return without running the model (Pi `@file`, `list files`).
pub fn infer_tool_calls_without_generation(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> Vec<ToolCall> {
    if !should_force_tool_call(messages, tools, tool_choice) {
        return Vec::new();
    }
    infer_tool_call_from_context(messages, tools)
        .into_iter()
        .collect()
}

const GEMMA4_STR: &str = "<|\"|>";

fn gemma4_dict_to_json_object(dict: &str) -> Option<Value> {
    let trimmed = dict.trim();
    if trimmed.is_empty() || trimmed == "{}" {
        return Some(serde_json::json!({}));
    }
    let (val, _rest) = parse_gemma4_value(trimmed)?;
    match val {
        Value::Object(_) => Some(val),
        _ => None,
    }
}

fn parse_gemma4_value(s: &str) -> Option<(Value, &str)> {
    let s = s.trim_start();
    if s.is_empty() {
        return None;
    }
    if s.starts_with(GEMMA4_STR) {
        let after = &s[GEMMA4_STR.len()..];
        let end = after.find(GEMMA4_STR)?;
        return Some((
            Value::String(after[..end].to_string()),
            &after[end + GEMMA4_STR.len()..],
        ));
    }
    if s.starts_with('"') {
        let (inner, consumed) = parse_json_string_lenient(s)?;
        return Some((Value::String(inner), &s[consumed..]));
    }
    if s.starts_with('{') {
        return parse_gemma4_object(s);
    }
    if s.starts_with('[') {
        return parse_gemma4_array(s);
    }
    if s.starts_with("true") && !ident_continues(&s[4..]) {
        return Some((Value::Bool(true), &s[4..]));
    }
    if s.starts_with("false") && !ident_continues(&s[5..]) {
        return Some((Value::Bool(false), &s[5..]));
    }
    if s.starts_with("null") && !ident_continues(&s[4..]) {
        return Some((Value::Null, &s[4..]));
    }
    if let Some((n, rest)) = parse_gemma4_number(s) {
        return Some((Value::Number(n), rest));
    }
    let end = s
        .find(|c: char| c == ',' || c == '}' || c == ']' || c.is_whitespace() || c == '<')
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((Value::String(s[..end].to_string()), &s[end..]))
}

fn ident_continues(s: &str) -> bool {
    s.chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn parse_json_string(s: &str) -> Option<(String, usize)> {
    parse_json_string_inner(s, false)
}

/// Gemma sometimes emits JSON strings inside native tool dicts with unescaped
/// quotes (`"eligible": true` inside `oldText`). Close a string only when `"`
/// is followed by a sibling key / container close, not by more prose.
fn parse_json_string_lenient(s: &str) -> Option<(String, usize)> {
    parse_json_string_inner(s, true)
}

fn skip_ws_bytes(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn is_lenient_json_string_end(bytes: &[u8], quote_idx: usize) -> bool {
    let i = skip_ws_bytes(bytes, quote_idx + 1);
    if i >= bytes.len() {
        return true;
    }
    match bytes[i] {
        b',' => {
            let j = skip_ws_bytes(bytes, i + 1);
            j >= bytes.len()
                || matches!(bytes[j], b'"' | b'}' | b']' | b'_')
                || bytes[j].is_ascii_alphabetic()
        }
        b'}' | b']' => {
            let j = skip_ws_bytes(bytes, i + 1);
            j >= bytes.len() || matches!(bytes[j], b',' | b'}' | b']' | b'<')
        }
        _ => false,
    }
}

fn parse_json_string_inner(s: &str, lenient: bool) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = 1usize;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escape {
            match b {
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                other => out.push(other as char),
            }
            escape = false;
        } else if b == b'\\' {
            escape = true;
        } else if b == b'"' {
            if lenient {
                if is_lenient_json_string_end(bytes, i) {
                    return Some((out, i + 1));
                }
                out.push('"');
            } else {
                return Some((out, i + 1));
            }
        } else {
            out.push(b as char);
        }
        i += 1;
    }
    if lenient {
        Some((out, bytes.len()))
    } else {
        None
    }
}

fn parse_gemma4_number(s: &str) -> Option<(serde_json::Number, &str)> {
    let bytes = s.as_bytes();
    let mut i = 0usize;
    if bytes.first() == Some(&b'-') {
        i = 1;
    }
    if i >= bytes.len() || !bytes[i].is_ascii_digit() {
        return None;
    }
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(bytes.get(i), Some(&b'e' | &b'E')) {
        i += 1;
        if matches!(bytes.get(i), Some(&b'+' | &b'-')) {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if ident_continues(&s[i..]) {
        return None;
    }
    let raw = &s[..i];
    let n = serde_json::from_str::<serde_json::Number>(raw).ok()?;
    Some((n, &s[i..]))
}

fn parse_gemma4_key(s: &str) -> Option<(String, &str)> {
    let s = s.trim_start();
    if s.starts_with(GEMMA4_STR) {
        let after = &s[GEMMA4_STR.len()..];
        let end = after.find(GEMMA4_STR)?;
        return Some((after[..end].to_string(), &after[end + GEMMA4_STR.len()..]));
    }
    if s.starts_with('"') {
        let (inner, consumed) = parse_json_string(s)?;
        return Some((inner, &s[consumed..]));
    }
    let end = s
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphanumeric() && *c != '_')
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    Some((s[..end].to_string(), &s[end..]))
}

fn parse_gemma4_object(s: &str) -> Option<(Value, &str)> {
    debug_assert!(s.starts_with('{'));
    let mut rest = s[1..].trim_start();
    let mut map = serde_json::Map::new();
    loop {
        rest = rest.trim_start();
        if rest.starts_with('}') {
            return Some((Value::Object(map), &rest[1..]));
        }
        if rest.is_empty() || rest.starts_with('<') {
            return Some((Value::Object(map), rest));
        }
        let (key, after_key) = parse_gemma4_key(rest)?;
        rest = after_key.trim_start();
        if !rest.starts_with(':') {
            return None;
        }
        rest = rest[1..].trim_start();
        let (val, after_val) = parse_gemma4_value(rest)?;
        map.insert(key, val);
        rest = after_val.trim_start();
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
            continue;
        }
        if rest.starts_with('}') {
            return Some((Value::Object(map), &rest[1..]));
        }
        if rest.is_empty() || rest.starts_with('<') {
            return Some((Value::Object(map), rest));
        }
        return None;
    }
}

fn parse_gemma4_array(s: &str) -> Option<(Value, &str)> {
    debug_assert!(s.starts_with('['));
    let mut rest = s[1..].trim_start();
    let mut items = Vec::new();
    loop {
        rest = rest.trim_start();
        if rest.starts_with(']') {
            return Some((Value::Array(items), &rest[1..]));
        }
        if rest.is_empty() || rest.starts_with('<') {
            return Some((Value::Array(items), rest));
        }
        let (val, after) = parse_gemma4_value(rest)?;
        items.push(val);
        rest = after.trim_start();
        if rest.starts_with(',') {
            rest = rest[1..].trim_start();
            continue;
        }
        if rest.starts_with(']') {
            return Some((Value::Array(items), &rest[1..]));
        }
        if rest.is_empty() || rest.starts_with('<') {
            return Some((Value::Array(items), rest));
        }
        return None;
    }
}

fn fill_default_tool_args(name: &str, arguments: &str) -> (String, String) {
    if name == "bash" {
        let trimmed = arguments.trim();
        let missing = trimmed.is_empty()
            || trimmed == "{}"
            || serde_json::from_str::<Value>(trimmed)
                .ok()
                .and_then(|v| v.get("command").and_then(|c| c.as_str()).map(|s| s.trim().is_empty()))
                .unwrap_or(false);
        if missing {
            return (name.to_string(), r#"{"command":"ls -F"}"#.to_string());
        }
    }
    (name.to_string(), arguments.to_string())
}

/// Map hallucinated tool names (e.g. `list_files`) onto declared tools.
fn normalize_tool_call(name: &str, arguments: &str, allowed: &[String]) -> (String, String) {
    if allowed.is_empty() || allowed.iter().any(|n| n == name) {
        return fill_default_tool_args(name, arguments);
    }
    let lower = name.to_lowercase().replace('-', "_");
    let args_empty = arguments.trim().is_empty() || arguments.trim() == "{}";
    if (lower.contains("list") || lower == "ls" || lower == "dir")
        && allowed.iter().any(|n| n == "bash")
    {
        let args = if args_empty {
            r#"{"command":"ls -F"}"#.to_string()
        } else {
            arguments.to_string()
        };
        return fill_default_tool_args("bash", &args);
    }
    if lower.contains("read") && allowed.iter().any(|n| n == "read") {
        return fill_default_tool_args("read", arguments);
    }
    if lower.contains("write") && allowed.iter().any(|n| n == "write") {
        return fill_default_tool_args("write", arguments);
    }
    if lower.contains("edit") && allowed.iter().any(|n| n == "edit") {
        return fill_default_tool_args("edit", arguments);
    }
    if (lower.contains("bash") || lower.contains("shell") || lower.contains("exec") || lower.contains("run"))
        && allowed.iter().any(|n| n == "bash")
    {
        return fill_default_tool_args("bash", arguments);
    }
    if let Some(best) = allowed.iter().find(|n| {
        let nl = n.to_lowercase();
        lower.starts_with(&nl) || nl.starts_with(&lower)
    }) {
        return fill_default_tool_args(best, arguments);
    }
    fill_default_tool_args(name, arguments)
}

fn finish_parse_tool_calls(calls: Vec<ToolCall>, allowed: Option<&[String]>) -> Vec<ToolCall> {
    match allowed {
        Some(names) if !names.is_empty() => calls
            .into_iter()
            .map(|mut tc| {
                let (name, args) =
                    normalize_tool_call(&tc.function.name, &tc.function.arguments, names);
                tc.function.name = name;
                tc.function.arguments = args;
                tc
            })
            .collect(),
        _ => calls,
    }
}

fn parse_native_tool_calls(text: &str, allowed: Option<&[String]>) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(NATIVE_TOOL_CALL_TRIGGER) {
        let after_trigger = &rest[start + NATIVE_TOOL_CALL_TRIGGER.len()..];
        let after = after_trigger
            .strip_prefix("call:")
            .or_else(|| after_trigger.strip_prefix("call"))
            .unwrap_or(after_trigger);
        let body = if let Some(end) = after.find(NATIVE_TOOL_CALL_SUFFIX) {
            &after[..end]
        } else {
            after.trim()
        };
        if let Some(brace) = body.find('{') {
            let name = body[..brace].trim();
            let dict = &body[brace..];
            if !name.is_empty() {
                if let Some(args) = gemma4_dict_to_json_object(dict) {
                    calls.push(make_tool_call(name, args));
                }
            }
        }
        let advance = if let Some(end) = after.find(NATIVE_TOOL_CALL_SUFFIX) {
            start + NATIVE_TOOL_CALL_TRIGGER.len() + end + NATIVE_TOOL_CALL_SUFFIX.len()
        } else {
            rest.len()
        };
        if advance <= start {
            break;
        }
        rest = &rest[advance.min(rest.len())..];
    }
    if calls.is_empty() {
        parse_bare_gemma4_call_invocations(text, &mut calls);
    }
    finish_parse_tool_calls(calls, allowed)
}

fn parse_bare_gemma4_call_invocations(text: &str, calls: &mut Vec<ToolCall>) {
    let mut rest = text;
    while let Some(idx) = rest.find("call:") {
        if idx > 0 {
            let prev = rest.as_bytes()[idx - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                rest = &rest[idx + 5..];
                continue;
            }
        }
        let after = &rest[idx + 5..];
        let Some(brace) = after.find('{') else {
            rest = &rest[idx + 5..];
            continue;
        };
        let name = after[..brace].trim();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            rest = &rest[idx + 5..];
            continue;
        }
        let dict_start = &after[brace..];
        let dict = if let Some(end) = dict_start.find(NATIVE_TOOL_CALL_SUFFIX) {
            &dict_start[..end]
        } else {
            dict_start.trim()
        };
        if let Some(args) = gemma4_dict_to_json_object(dict) {
            calls.push(make_tool_call(name, args));
        }
        let advance = idx + 5 + brace + dict.len().min(dict_start.len());
        if advance == 0 || advance > rest.len() {
            break;
        }
        rest = &rest[advance.min(rest.len())..];
    }
}

fn has_channel_markup(text: &str) -> bool {
    text.contains(CHANNEL_START) || text.contains(CHANNEL_END)
}

fn parse_named_channel_body(body: &str) -> (String, String) {
    if let Some(newline) = body.find('\n') {
        (
            body[..newline].trim().to_string(),
            body[newline + 1..].trim().to_string(),
        )
    } else {
        (body.trim().to_string(), String::new())
    }
}

fn join_visible_parts(parts: &[String]) -> String {
    parts
        .iter()
        .map(String::as_str)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Oracle `split_channel_markup`: `<|channel>thought` is reasoning, `<|channel>final`
/// (or text after a closed thought) is visible content.
fn split_channel_markup(text: &str) -> (String, String) {
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    let mut rest = text;
    let mut after_explicit_thought = false;

    loop {
        if let Some(start) = rest.find(CHANNEL_START) {
            if start > 0 {
                let segment = rest[..start].trim();
                if !segment.is_empty() {
                    if after_explicit_thought {
                        content_parts.push(segment.to_string());
                        after_explicit_thought = false;
                    } else {
                        reasoning_parts.push(segment.to_string());
                    }
                }
            }
            rest = &rest[start + CHANNEL_START.len()..];
            if let Some(end) = rest.find(CHANNEL_END) {
                let body = &rest[..end];
                rest = &rest[end + CHANNEL_END.len()..];
                let (name, body_text) = parse_named_channel_body(body);
                match name.as_str() {
                    "final" => {
                        after_explicit_thought = false;
                        if !body_text.is_empty() {
                            content_parts.push(body_text);
                        }
                    }
                    "thought" => {
                        after_explicit_thought = true;
                        if !body_text.is_empty() {
                            reasoning_parts.push(body_text);
                        }
                    }
                    _ => {
                        after_explicit_thought = false;
                        if !body_text.is_empty() {
                            reasoning_parts.push(body_text);
                        }
                    }
                }
                continue;
            }
            let (name, body_text) = parse_named_channel_body(rest);
            match name.as_str() {
                "final" => {
                    if !body_text.is_empty() {
                        content_parts.push(body_text);
                    }
                }
                "thought" => {
                    if !body_text.is_empty() {
                        reasoning_parts.push(body_text);
                    }
                }
                _ => {
                    if !body_text.is_empty() {
                        reasoning_parts.push(body_text);
                    }
                }
            }
            break;
        }

        if let Some(end) = rest.find(CHANNEL_END) {
            let segment = rest[..end].trim();
            if !segment.is_empty() {
                reasoning_parts.push(segment.to_string());
            }
            rest = &rest[end + CHANNEL_END.len()..];
            after_explicit_thought = false;
            continue;
        }

        let segment = rest.trim();
        if !segment.is_empty() {
            if after_explicit_thought {
                content_parts.push(segment.to_string());
            } else if content_parts.is_empty() {
                reasoning_parts.push(segment.to_string());
            } else {
                content_parts.push(segment.to_string());
            }
        }
        break;
    }

    (join_visible_parts(&reasoning_parts), join_visible_parts(&content_parts))
}

fn strip_native_tool_calls(text: &mut String) {
    while let Some(start) = text.find(NATIVE_TOOL_CALL_TRIGGER) {
        let after = &text[start + NATIVE_TOOL_CALL_TRIGGER.len()..];
        let rel_end = if let Some(end) = after.find(NATIVE_TOOL_CALL_SUFFIX) {
            NATIVE_TOOL_CALL_TRIGGER.len() + end + NATIVE_TOOL_CALL_SUFFIX.len()
        } else {
            // Generation often stops on `<tool_call|>`, so the suffix is absent.
            text.len() - start
        };
        text.replace_range(start..start + rel_end, "");
    }
    *text = text.trim().to_string();
}

fn find_paragraph_answer_start(text: &str) -> Option<usize> {
    const PLANNING_PREFIXES: &[&str] = &[
        "User ",
        "The user ",
        "I need ",
        "I will ",
        "I should ",
        "Let me ",
        "**Analysis",
        "Analysis",
        "Okay",
        "Ok,",
        "First,",
        "Step ",
    ];
    let mut best: Option<usize> = None;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i + 2 <= bytes.len() {
        if bytes[i] == b'\n' && bytes[i + 1] == b'\n' {
            let start = i + 2;
            let rest = &text[start..];
            let para = rest.lines().next().unwrap_or("").trim();
            if !para.is_empty()
                && !PLANNING_PREFIXES.iter().any(|p| para.starts_with(p))
            {
                best = Some(start);
            }
            i = start;
            continue;
        }
        i += 1;
    }
    best
}

/// Split Gemma 4 generation into (reasoning, visible content). Tool-call markup
/// stays in `text` for [`resolve_tool_calls`]; it is stripped from content.
pub fn split_reasoning_and_content(text: &str) -> (String, String) {
    if !has_channel_markup(text) {
        let mut content = text.to_string();
        strip_native_tool_calls(&mut content);
        return (String::new(), content);
    }
    let (mut reasoning, mut content) = split_channel_markup(text);
    if content.is_empty() {
        if let Some(idx) = find_paragraph_answer_start(&reasoning) {
            let answer = reasoning[idx..].trim().to_string();
            let kept = reasoning[..idx].trim().to_string();
            if !answer.is_empty() {
                reasoning = kept;
                content = answer;
            }
        }
    }
    strip_native_tool_calls(&mut content);
    (reasoning, content)
}

/// Parse native Gemma 4 tool calls from generated text, then Pi inference fallback.
pub fn resolve_tool_calls(
    text: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> Vec<ToolCall> {
    let allowed: Option<Vec<String>> =
        tools.map(|t| t.iter().map(|x| x.function.name.clone()).collect());
    let parsed = parse_native_tool_calls(text, allowed.as_deref());
    if !parsed.is_empty() {
        return parsed;
    }
    infer_tool_calls_without_generation(messages, tools, tool_choice)
}

pub fn apply_chat_template(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> String {
    let mut prompt = String::new();
    let mut system_parts = Vec::new();
    for msg in messages {
        if msg.role == "system" {
            if let Some(content) = &msg.content {
                if !content.is_empty() {
                    system_parts.push(content.clone());
                }
            }
        }
    }
    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    let include_tool_declarations = has_tools && awaits_tool_call(messages);
    if include_tool_declarations || !system_parts.is_empty() {
        prompt.push_str(TURN_START);
        prompt.push_str("system\n");
        prompt.push_str("<|think|>\n");
        if !system_parts.is_empty() {
            prompt.push_str(&system_parts.join("\n\n"));
            if include_tool_declarations {
                prompt.push('\n');
            }
        }
        if let Some(tools) = tools {
            if include_tool_declarations {
                prompt.push_str(&render_tool_declarations(tools));
            }
        }
        if should_force_tool_call(messages, tools, tool_choice) {
            prompt.push_str("\nYou must call a tool to answer this request.");
        }
        prompt.push_str(TURN_END);
        prompt.push('\n');
    }

    let mut ends_in_open_model_turn = false;
    let mut i = 0;
    while i < messages.len() {
        let msg = &messages[i];
        if msg.role == "system" {
            i += 1;
            continue;
        }
        if msg.role == "tool" {
            let name = msg.name.as_deref().unwrap_or("tool");
            prompt.push_str(&render_tool_response(
                name,
                msg.content.as_deref().unwrap_or(""),
            ));
            i += 1;
            continue;
        }
        if msg.role == "user" {
            ends_in_open_model_turn = false;
        }
        let mapped = if msg.role == "assistant" { "model" } else { "user" };
        prompt.push_str(TURN_START);
        prompt.push_str(mapped);
        prompt.push('\n');
        if let Some(content) = &msg.content {
            prompt.push_str(content);
        }
        if let Some(tool_calls) = &msg.tool_calls {
            for tc in tool_calls {
                prompt.push_str(&render_assistant_tool_call(tc));
            }
        }
        i += 1;
        if msg.role == "assistant" && msg.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) {
            ends_in_open_model_turn = true;
            while i < messages.len() && messages[i].role == "tool" {
                let tm = &messages[i];
                let name = tm.name.as_deref().unwrap_or("tool");
                prompt.push_str(&render_tool_response(
                    name,
                    tm.content.as_deref().unwrap_or(""),
                ));
                i += 1;
            }
            while i < messages.len()
                && messages[i].role == "assistant"
                && messages[i]
                    .tool_calls
                    .as_ref()
                    .map_or(true, |t| t.is_empty())
            {
                if let Some(content) = &messages[i].content {
                    prompt.push_str(content);
                }
                ends_in_open_model_turn = false;
                i += 1;
            }
        }
        if !ends_in_open_model_turn {
            prompt.push_str(TURN_END);
            prompt.push('\n');
        }
    }

    if ends_in_open_model_turn {
        prompt.push_str(generation_priming_suffix(messages, tools, tool_choice));
    } else {
        prompt.push_str(TURN_START);
        prompt.push_str("model\n");
        prompt.push_str(generation_priming_suffix(messages, tools, tool_choice));
    }
    prompt
}

fn generation_priming_suffix(
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    tool_choice: Option<&Value>,
) -> &'static str {
    // Match metal-llm-server: only prime native tool-call markup when the
    // client explicitly requires a tool (`tool_choice=required`).
    if tool_choice_is_required(tool_choice) && should_force_tool_call(messages, tools, tool_choice) {
        NATIVE_TOOL_CALL_TRIGGER
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_tool() -> Tool {
        Tool {
            tool_type: Some("function".to_string()),
            function: FunctionDef {
                name: "read".to_string(),
                description: Some("Read a file".to_string()),
                parameters: None,
            },
        }
    }

    #[test]
    fn template_puts_think_and_tools_on_system() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: Some("be helpful".into()),
                ..Default::default()
            },
            ChatMessage {
                role: "user".into(),
                content: Some("hi".into()),
                ..Default::default()
            },
        ];
        let tools = vec![read_tool()];
        let prompt = apply_chat_template(&messages, Some(&tools), None);
        assert!(prompt.contains("<|turn>system\n<|think|>\nbe helpful\n"));
        assert!(prompt.contains("<|tool>declaration:read"));
        assert!(prompt.contains("You must call a tool to answer this request."));
        assert!(prompt.ends_with("<|turn>model\n"));
        assert!(!prompt.contains("<|tool_call>"));
    }

    #[test]
    fn splits_thought_channel_from_final() {
        let text = "<|channel>thought\nI should read the file.\n<channel|>\n<|channel>final\nIt is a Metal LLM server.\n<channel|>";
        let (r, c) = split_reasoning_and_content(text);
        assert!(r.contains("I should read"));
        assert_eq!(c, "It is a Metal LLM server.");
    }

    #[test]
    fn splits_answer_after_closed_thought() {
        let text = "<|channel>thought\nThe user asked about the project.\n<channel|>\nThis is a Failed Experiments Log for Metal inference.";
        let (r, c) = split_reasoning_and_content(text);
        assert!(r.contains("The user asked"));
        assert!(c.contains("Failed Experiments Log"));
    }

    #[test]
    fn infers_read_for_pi_at_file() {
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some("whats this project about @AGENTS.md".into()),
            ..Default::default()
        }];
        let tools = vec![read_tool()];
        let calls = infer_tool_calls_without_generation(&messages, Some(&tools), None);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"AGENTS.md"}"#);
    }

    #[test]
    fn parses_native_read_call() {
        let text = r#"<|tool_call>call:read{path:<|"|>AGENTS.md<|"|>}<tool_call|>"#;
        let calls = parse_native_tool_calls(text, Some(&["read".to_string()]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"AGENTS.md"}"#);
    }

    #[test]
    fn parses_native_edit_with_nested_edits_array() {
        let text = r#"<|tool_call>call:edit{edits:[{oldText:<|"|>Change the JSON type from string to integer.<|"|>,newText:<|"|>Keep it a string like "89".<|"|>}],path:<|"|>demos/max_age_cieiling_proposal.md<|"|>}"#;
        let allowed = vec![
            "read".to_string(),
            "edit".to_string(),
            "write".to_string(),
            "bash".to_string(),
        ];
        let calls = parse_native_tool_calls(text, Some(&allowed));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args["path"].as_str(),
            Some("demos/max_age_cieiling_proposal.md")
        );
        let edits = args["edits"].as_array().expect("edits array");
        assert_eq!(edits.len(), 1);
        assert_eq!(
            edits[0]["oldText"].as_str(),
            Some("Change the JSON type from string to integer.")
        );
        assert_eq!(
            edits[0]["newText"].as_str(),
            Some(r#"Keep it a string like "89"."#)
        );
    }

    #[test]
    fn parses_native_call_without_suffix_stop() {
        let text = r#"<|channel>thought
I should edit the file.
<channel|><|tool_call>call:edit{edits:[{oldText:<|"|>integer<|"|>,newText:<|"|>"89"<|"|>}],path:<|"|>demos/max_age_cieiling_proposal.md<|"|>}"#;
        let calls = parse_native_tool_calls(text, Some(&["edit".to_string()]));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        let (_, content) = split_reasoning_and_content(text);
        assert!(!content.contains("<|tool_call>"));
        assert!(!content.contains("oldText"));
    }

    #[test]
    fn parses_json_style_edit_args_with_unescaped_quotes() {
        let text = r#"<|channel>thought
The user wants a string ceiling.
<channel|><|tool_call>call:edit{edits:[{
  "oldText": "Change the JSON type from string to integer.
json{
 "eligible": true,
 "max_age_cieling": ""
}
The type today: string",
  "newText": "Keep max_age_cieiling a JSON string ("89"), not integer 89."
}],path:"demos/max_age_cieiling_proposal.md"}"#;
        let calls = parse_native_tool_calls(
            text,
            Some(&["read".to_string(), "edit".to_string()]),
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "edit");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(
            args["path"].as_str(),
            Some("demos/max_age_cieiling_proposal.md")
        );
        let edits = args["edits"].as_array().expect("edits");
        assert!(edits[0]["oldText"]
            .as_str()
            .unwrap()
            .contains(r#""eligible": true"#));
        assert!(edits[0]["newText"]
            .as_str()
            .unwrap()
            .contains(r#"("89")"#));

        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("fix demos/max_age_cieiling_proposal.md".into()),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                tool_calls: Some(vec![make_tool_call(
                    "read",
                    serde_json::json!({ "path": "demos/max_age_cieiling_proposal.md" }),
                )]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                name: Some("read".into()),
                content: Some("# RFC\nmax_age_cieiling is unused".into()),
                ..Default::default()
            },
        ];
        let tools = vec![Tool {
            tool_type: Some("function".into()),
            function: FunctionDef {
                name: "edit".into(),
                description: None,
                parameters: None,
            },
        }];
        let resolved = resolve_tool_calls(text, &messages, Some(&tools), None);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].function.name, "edit");
        assert!(!awaits_tool_call(&messages));
    }

    #[test]
    fn remaps_list_files_to_bash() {
        let text = r#"<|tool_call>call:list_files{}<tool_call|>"#;
        let allowed = vec!["read".to_string(), "bash".to_string(), "edit".to_string()];
        let calls = parse_native_tool_calls(text, Some(&allowed));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "bash");
        assert_eq!(calls[0].function.arguments, r#"{"command":"ls -F"}"#);
    }

    #[test]
    fn tool_result_stays_in_open_model_turn() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("whats this @AGENTS.md".into()),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                tool_calls: Some(vec![make_tool_call(
                    "read",
                    serde_json::json!({ "path": "AGENTS.md" }),
                )]),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                name: Some("read".into()),
                content: Some("# ksearch\ncompiler".into()),
                ..Default::default()
            },
        ];
        let prompt = apply_chat_template(&messages, None, None);
        assert!(prompt.contains("<|tool_response>response:read"));
        assert!(prompt.contains("compiler"));
        assert!(prompt.contains("<|turn>model\n"));
    }
}
