/// Tokenizes text into lowercase alphanumeric and symbol terms, matching regex r"\b[a-zA-Z0-9_\-\.\$]+\b".
#[inline]
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut start = None;

    for i in 0..len {
        let b = bytes[i];
        let is_word_char = b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' || b == b'$';
        if is_word_char {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start {
            let slice = &text[s..i];
            tokens.push(slice.to_lowercase());
            start = None;
        }
    }

    if let Some(s) = start {
        tokens.push(text[s..len].to_lowercase());
    }

    tokens
}
