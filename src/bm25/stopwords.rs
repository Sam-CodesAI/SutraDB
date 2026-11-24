use std::collections::HashSet;
use std::sync::OnceLock;

static ENGLISH_STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static SPANISH_STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static FRENCH_STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static GERMAN_STOPWORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn get_english() -> &'static HashSet<&'static str> {
    ENGLISH_STOPWORDS.get_or_init(|| {
        [
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves"
        ].into_iter().collect()
    })
}

fn get_spanish() -> &'static HashSet<&'static str> {
    SPANISH_STOPWORDS.get_or_init(|| {
        [
            "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
            "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo",
            "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque",
            "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
            "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni",
            "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes",
            "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa"
        ].into_iter().collect()
    })
}

fn get_french() -> &'static HashSet<&'static str> {
    FRENCH_STOPWORDS.get_or_init(|| {
        [
            "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
            "que", "est", "pour", "qui", "dans", "a", "par", "plus", "au", "avec",
            "ce", "ne", "pas", "sur", "se", "sont", "ont", "aux", "il", "son",
            "sa", "ses", "mais", "comme", "ou", "nous", "vous", "ils", "cette", "leur",
            "tout", "aussi", "ces", "lui", "elle", "y", "sans", "si", "bien", "fait"
        ].into_iter().collect()
    })
}

fn get_german() -> &'static HashSet<&'static str> {
    GERMAN_STOPWORDS.get_or_init(|| {
        [
            "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als",
            "auch", "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach",
            "wird", "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
            "einen", "so", "zum", "war", "haben", "nur", "oder", "aber", "vor", "zur"
        ].into_iter().collect()
    })
}

/// Checks whether a given lowercase term is a stopword in the target language.
pub fn is_stopword(word: &str, lang: &str) -> bool {
    match lang.to_lowercase().as_str() {
        "spanish" | "es" => get_spanish().contains(word),
        "french" | "fr" => get_french().contains(word),
        "german" | "de" => get_german().contains(word),
        _ => get_english().contains(word),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopwords() {
        assert!(is_stopword("the", "english"));
        assert!(is_stopword("and", "english"));
        assert!(!is_stopword("kubernetes", "english"));

        assert!(is_stopword("de", "spanish"));
        assert!(is_stopword("por", "spanish"));
        assert!(!is_stopword("servidor", "spanish"));

        assert!(is_stopword("les", "french"));
        assert!(is_stopword("pour", "french"));

        assert!(is_stopword("und", "german"));
        assert!(is_stopword("der", "german"));
    }
}
