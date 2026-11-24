use serde::{Deserialize, Serialize};
use super::stemmer::stem;
use super::stopwords::is_stopword;

/// Configuration options for the text analyzer pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    pub lowercase: bool,
    pub stemmer: bool,
    pub stopwords: bool,
    pub language: String,
    pub split_identifiers: bool,
    pub cjk_bigrams: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            stemmer: true,
            stopwords: true,
            language: "english".to_string(),
            split_identifiers: true,
            cjk_bigrams: true,
        }
    }
}

/// Text analyzer orchestrating tokenization, script normalization, identifier splitting,
/// stopword filtering, and linguistic stemming.
#[derive(Debug, Clone)]
pub struct Analyzer {
    pub config: AnalyzerConfig,
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new(AnalyzerConfig::default())
    }
}

impl Analyzer {
    pub fn new(config: AnalyzerConfig) -> Self {
        Self { config }
    }

    /// Analyzes a raw string and outputs a normalized, stemmed token sequence.
    pub fn analyze(&self, text: &str) -> Vec<String> {
        if text.trim().is_empty() {
            return Vec::new();
        }

        let mut raw_tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            let ch = chars[i];

            // 1. CJK Character Handling
            if self.config.cjk_bigrams && is_cjk(ch) {
                let start = i;
                while i < len && is_cjk(chars[i]) {
                    i += 1;
                }
                let cjk_slice: String = chars[start..i].iter().collect();
                raw_tokens.extend(generate_cjk_tokens(&cjk_slice));
                continue;
            }

            // 2. Alphanumeric / Identifier Handling
            let is_word_char = ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' || ch == '$';
            if is_word_char {
                let start = i;
                while i < len {
                    let c = chars[i];
                    if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == '$' {
                        i += 1;
                    } else {
                        break;
                    }
                }
                let token: String = chars[start..i].iter().collect();
                if self.config.split_identifiers && contains_camel_or_snake(&token) {
                    let sub_tokens = split_identifier(&token);
                    raw_tokens.extend(sub_tokens);
                } else {
                    raw_tokens.push(token);
                }
                continue;
            }

            i += 1;
        }

        // 3. Normalization, Stopwords & Stemming Pipeline
        let mut result = Vec::with_capacity(raw_tokens.len());
        for t in raw_tokens {
            let mut tok = if self.config.lowercase {
                t.to_lowercase()
            } else {
                t
            };

            // Trim punctuation edges (e.g. trailing period or leading dash)
            tok = tok.trim_matches(|c: char| c == '.' || c == '-' || c == '_').to_string();
            if tok.is_empty() {
                continue;
            }

            // Stopword pruning
            if self.config.stopwords && is_stopword(&tok, &self.config.language) {
                continue;
            }

            // Linguistic Stemming
            if self.config.stemmer {
                tok = stem(&tok, &self.config.language);
            }

            result.push(tok);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// CJK Unicode Range Detection
// ---------------------------------------------------------------------------

#[inline]
pub fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Extension A
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

fn generate_cjk_tokens(cjk_str: &str) -> Vec<String> {
    let chars: Vec<char> = cjk_str.chars().collect();
    let mut tokens = Vec::new();
    
    // Unigrams
    for &c in &chars {
        tokens.push(c.to_string());
    }

    // Bi-grams
    if chars.len() >= 2 {
        for window in chars.windows(2) {
            tokens.push(window.iter().collect());
        }
    }

    tokens
}

// ---------------------------------------------------------------------------
// Identifier Sub-word Splitting (camelCase & snake_case)
// ---------------------------------------------------------------------------

fn contains_camel_or_snake(s: &str) -> bool {
    if s.contains('_') || s.contains('-') {
        return true;
    }
    let chars: Vec<char> = s.chars().collect();
    for window in chars.windows(2) {
        if window[0].is_lowercase() && window[1].is_uppercase() {
            return true;
        }
    }
    false
}

pub fn split_identifier(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    // Always include full original token
    tokens.push(s.to_string());

    // Split on delimiters '_' and '-'
    let parts: Vec<&str> = s.split(|c| c == '_' || c == '-').filter(|p| !p.is_empty()).collect();
    for part in parts {
        let chars: Vec<char> = part.chars().collect();
        if chars.is_empty() {
            continue;
        }

        let mut start = 0;
        let len = chars.len();
        for i in 1..len {
            let prev = chars[i - 1];
            let curr = chars[i];
            // Split lowercase followed by uppercase: fooBar -> foo, Bar
            let lower_to_upper = prev.is_lowercase() && curr.is_uppercase();
            // Split letter to digit or digit to letter: v3 -> v, 3
            let alpha_digit_trans = (prev.is_alphabetic() && curr.is_numeric()) || (prev.is_numeric() && curr.is_alphabetic());

            if lower_to_upper || alpha_digit_trans {
                let sub: String = chars[start..i].iter().collect();
                if !sub.is_empty() {
                    tokens.push(sub);
                }
                start = i;
            }
        }
        if start < len {
            let sub: String = chars[start..len].iter().collect();
            if !sub.is_empty() {
                tokens.push(sub);
            }
        }
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_stem_and_stopwords() {
        let analyzer = Analyzer::default();
        let tokens = analyzer.analyze("Deploying microservices and connection pooling in Kubernetes");
        // "and" & "in" are pruned as stopwords
        // "deploying" -> "deploi"
        // "microservices" -> "microservic"
        // "connection" -> "connect"
        // "pooling" -> "pool"
        // "kubernetes" -> "kubernet"
        assert_eq!(tokens, vec!["deploi", "microservic", "connect", "pool", "kubernet"]);
    }

    #[test]
    fn test_analyzer_code_identifiers() {
        let analyzer = Analyzer::default();
        let tokens = analyzer.analyze("getUserProfile and fetch_auth_token");
        // "getUserProfile" emits "getuserprofile", "get", "user", "profile"
        // "and" is stopword dropped
        // "fetch_auth_token" emits "fetch_auth_token", "fetch", "auth", "token"
        assert!(tokens.contains(&"get".to_string()));
        assert!(tokens.contains(&"user".to_string()));
        assert!(tokens.contains(&"profil".to_string()));
        assert!(tokens.contains(&"fetch".to_string()));
        assert!(tokens.contains(&"token".to_string()));
    }

    #[test]
    fn test_analyzer_cjk() {
        let analyzer = Analyzer::default();
        let tokens = analyzer.analyze("自然语言处理");
        // Check unigram and bigram presence
        assert!(tokens.contains(&"自然".to_string()));
        assert!(tokens.contains(&"然语".to_string()));
        assert!(tokens.contains(&"语言".to_string()));
        assert!(tokens.contains(&"处理".to_string()));
    }
}
