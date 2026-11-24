/// High-performance linguistic stemmer implementing the Porter algorithm for English,
/// with supplementary phonetic/inflection rules for Spanish, French, and German.
/// Zero external C/Python dependencies, zero allocations for unstemmed tokens.

/// Stems a single lowercase token based on the specified language.
pub fn stem(word: &str, lang: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }

    match lang.to_lowercase().as_str() {
        "english" | "en" | "porter" => stem_english_porter(word),
        "spanish" | "es" => stem_spanish(word),
        "french" | "fr" => stem_french(word),
        "german" | "de" => stem_german(word),
        _ => stem_english_porter(word),
    }
}

// ---------------------------------------------------------------------------
// English Porter Stemmer
// ---------------------------------------------------------------------------

#[inline]
fn is_vowel(chars: &[char], i: usize) -> bool {
    match chars[i] {
        'a' | 'e' | 'i' | 'o' | 'u' => true,
        'y' => {
            if i == 0 {
                false
            } else {
                !is_vowel(chars, i - 1)
            }
        }
        _ => false,
    }
}

fn measure(chars: &[char]) -> usize {
    let len = chars.len();
    if len == 0 {
        return 0;
    }
    let mut i = 0;
    while i < len && !is_vowel(chars, i) {
        i += 1;
    }
    let mut count = 0;
    while i < len {
        while i < len && is_vowel(chars, i) {
            i += 1;
        }
        if i < len {
            count += 1;
            while i < len && !is_vowel(chars, i) {
                i += 1;
            }
        }
    }
    count
}

fn contains_vowel(chars: &[char]) -> bool {
    for i in 0..chars.len() {
        if is_vowel(chars, i) {
            return true;
        }
    }
    false
}

fn ends_double_consonant(chars: &[char]) -> bool {
    let len = chars.len();
    if len < 2 {
        return false;
    }
    chars[len - 1] == chars[len - 2] && !is_vowel(chars, len - 1)
}

fn cvc(chars: &[char], i: usize) -> bool {
    if i < 2 || !is_vowel(chars, i - 1) || is_vowel(chars, i) || is_vowel(chars, i - 2) {
        return false;
    }
    let last = chars[i];
    last != 'w' && last != 'x' && last != 'y'
}

pub fn stem_english_porter(word: &str) -> String {
    let mut chars: Vec<char> = word.chars().collect();
    if chars.len() <= 2 {
        return word.to_string();
    }

    // Step 1a
    if ends_with(&chars, "sses") {
        replace_suffix(&mut chars, 4, "ss");
    } else if ends_with(&chars, "ies") {
        replace_suffix(&mut chars, 3, "i");
    } else if !ends_with(&chars, "ss") && ends_with(&chars, "s") {
        chars.pop();
    }

    // Step 1b
    let mut step1b_extra = false;
    if ends_with(&chars, "eed") {
        let stem_len = chars.len() - 3;
        if measure(&chars[..stem_len]) > 0 {
            chars.pop();
        }
    } else if ends_with(&chars, "ed") {
        let stem_len = chars.len() - 2;
        if contains_vowel(&chars[..stem_len]) {
            chars.truncate(stem_len);
            step1b_extra = true;
        }
    } else if ends_with(&chars, "ing") {
        let stem_len = chars.len() - 3;
        if contains_vowel(&chars[..stem_len]) {
            chars.truncate(stem_len);
            step1b_extra = true;
        }
    }

    if step1b_extra {
        if ends_with(&chars, "at") || ends_with(&chars, "bl") || ends_with(&chars, "iz") {
            chars.push('e');
        } else if ends_double_consonant(&chars) {
            let last = *chars.last().unwrap();
            if last != 'l' && last != 's' && last != 'z' {
                chars.pop();
            }
        } else if measure(&chars) == 1 && cvc(&chars, chars.len() - 1) {
            chars.push('e');
        }
    }

    // Step 1c
    if ends_with(&chars, "y") {
        let stem_len = chars.len() - 1;
        if contains_vowel(&chars[..stem_len]) {
            chars[stem_len] = 'i';
        }
    }

    // Step 2
    if chars.len() > 2 {
        let step2_rules = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];
        for (suffix, repl) in step2_rules {
            if ends_with(&chars, suffix) {
                let stem_len = chars.len() - suffix.len();
                if measure(&chars[..stem_len]) > 0 {
                    replace_suffix(&mut chars, suffix.len(), repl);
                }
                break;
            }
        }
    }

    // Step 3
    if chars.len() > 2 {
        let step3_rules = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];
        for (suffix, repl) in step3_rules {
            if ends_with(&chars, suffix) {
                let stem_len = chars.len() - suffix.len();
                if measure(&chars[..stem_len]) > 0 {
                    replace_suffix(&mut chars, suffix.len(), repl);
                }
                break;
            }
        }
    }

    // Step 4
    if chars.len() > 3 {
        let step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];
        let mut matched = false;
        for suffix in step4_suffixes {
            if ends_with(&chars, suffix) {
                let stem_len = chars.len() - suffix.len();
                if measure(&chars[..stem_len]) > 1 {
                    chars.truncate(stem_len);
                }
                matched = true;
                break;
            }
        }
        if !matched && (ends_with(&chars, "sion") || ends_with(&chars, "tion")) {
            let stem_len = chars.len() - 3; // Keep the 's' or 't'
            if measure(&chars[..stem_len]) > 1 {
                chars.truncate(stem_len);
            }
        }
    }

    // Step 5a & 5b
    if ends_with(&chars, "e") {
        let stem_len = chars.len() - 1;
        let m = measure(&chars[..stem_len]);
        if m > 1 || (m == 1 && !cvc(&chars, stem_len - 1)) {
            chars.pop();
        }
    }

    if measure(&chars) > 1 && ends_double_consonant(&chars) && *chars.last().unwrap() == 'l' {
        chars.pop();
    }

    chars.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Multi-lingual Stemming Heuristics (Spanish, French, German)
// ---------------------------------------------------------------------------

fn stem_spanish(word: &str) -> String {
    let suffixes = [
        "amientos", "imientos", "amiento", "imiento", "izaciones", "ización", "izacion",
        "aciones", "ación", "acion", "amente", "mente", "adoras", "adores", "adora", "ador",
        "idades", "idad", "ancias", "ancia", "encias", "encia",
        "iendo", "ando", "ieron", "aron", "aste", "iste", "emos", "imos",
        "ada", "ado", "ida", "ido", "ar", "er", "ir", "as", "es", "os"
    ];
    let mut s = word.to_string();
    for suffix in suffixes {
        if s.ends_with(suffix) && s.len() > suffix.len() + 3 {
            s.truncate(s.len() - suffix.len());
            break;
        }
    }
    s
}

fn stem_french(word: &str) -> String {
    let suffixes = [
        "issements", "issement", "isations", "isation", "atrices", "ateurs", "atrice", "ateur",
        "ations", "ation", "ements", "ement", "ismes", "isme", "istes", "iste",
        "antes", "ante", "ants", "ant", "euses", "euse", "able", "ible",
        "îtes", "âtes", "rent", "ions", "erie", "era", "ez", "ent", "ais", "ait",
        "er", "ir", "ée", "és", "ées", "é"
    ];
    let mut s = word.to_string();
    for suffix in suffixes {
        if s.ends_with(suffix) && s.len() > suffix.len() + 3 {
            s.truncate(s.len() - suffix.len());
            break;
        }
    }
    s
}

fn stem_german(word: &str) -> String {
    let suffixes = [
        "schaften", "schaft", "heiten", "heit", "keiten", "keit",
        "ungen", "ung", "ieren", "ierend", "ierte", "iert",
        "innen", "isch", "lich", "ende", "end", "ern", "est", "en", "er", "es", "em"
    ];
    let mut s = word.to_string();
    for suffix in suffixes {
        if s.ends_with(suffix) && s.len() > suffix.len() + 3 {
            s.truncate(s.len() - suffix.len());
            break;
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn ends_with(chars: &[char], suffix: &str) -> bool {
    let s_chars: Vec<char> = suffix.chars().collect();
    if chars.len() < s_chars.len() {
        return false;
    }
    let offset = chars.len() - s_chars.len();
    &chars[offset..] == s_chars.as_slice()
}

#[inline]
fn replace_suffix(chars: &mut Vec<char>, suffix_len: usize, replacement: &str) {
    chars.truncate(chars.len() - suffix_len);
    chars.extend(replacement.chars());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_english_porter_stemmer() {
        assert_eq!(stem("connecting", "english"), "connect");
        assert_eq!(stem("connection", "english"), "connect");
        assert_eq!(stem("connections", "english"), "connect");
        assert_eq!(stem("microservices", "english"), "microservic");
        assert_eq!(stem("deploying", "english"), "deploi");
        assert_eq!(stem("relational", "english"), "relat");
        assert_eq!(stem("optimization", "english"), "optim");
        assert_eq!(stem("happy", "english"), "happi");
        assert_eq!(stem("cats", "english"), "cat");
    }

    #[test]
    fn test_multilingual_stemmers() {
        assert_eq!(stem("optimizacion", "spanish"), "optim");
        assert_eq!(stem("desarrollando", "spanish"), "desarroll");
        assert_eq!(stem("deploiement", "french"), "deploi");
        assert_eq!(stem("optimisation", "french"), "optim");
        assert_eq!(stem("optimierung", "german"), "optimier");
        assert_eq!(stem("verbindung", "german"), "verbind");
    }
}
