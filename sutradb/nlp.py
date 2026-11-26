"""
Linguistic text analyzer pipeline for SutraDB BM25 hybrid search.
Provides Porter stemming, multi-lingual stopword removal, camelCase/snake_case
identifier splitting, and CJK (Chinese, Japanese, Korean) bigram sliding windows.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Multi-lingual Stopwords
# ---------------------------------------------------------------------------

ENGLISH_STOPWORDS: Set[str] = {
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
}

SPANISH_STOPWORDS: Set[str] = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra"
}

FRENCH_STOPWORDS: Set[str] = {
    "de", "la", "le", "et", "les", "des", "en", "un", "du", "une", "que", "est",
    "pour", "qui", "dans", "a", "par", "plus", "au", "avec", "ce", "ne", "pas",
    "sur", "se", "sont", "ont", "aux", "il", "son", "sa", "ses", "mais", "comme"
}

GERMAN_STOPWORDS: Set[str] = {
    "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des",
    "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch",
    "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach", "wird", "bei"
}

STOPWORDS: Set[str] = ENGLISH_STOPWORDS


def get_stopwords(language: str = "english") -> Set[str]:
    lang = language.lower()
    if lang in ("spanish", "es"):
        return SPANISH_STOPWORDS
    elif lang in ("french", "fr"):
        return FRENCH_STOPWORDS
    elif lang in ("german", "de"):
        return GERMAN_STOPWORDS
    return ENGLISH_STOPWORDS

# ---------------------------------------------------------------------------
# Porter Stemmer
# ---------------------------------------------------------------------------

class PorterStemmer:
    """Standard algorithmic Porter Stemmer for English text."""

    @staticmethod
    def _is_vowel(word: str, i: int) -> bool:
        ch = word[i]
        if ch in "aeiou":
            return True
        if ch == "y":
            return i != 0 and not PorterStemmer._is_vowel(word, i - 1)
        return False

    @staticmethod
    def _measure(word: str) -> int:
        n = 0
        i = 0
        length = len(word)
        while i < length and not PorterStemmer._is_vowel(word, i):
            i += 1
        while i < length:
            while i < length and PorterStemmer._is_vowel(word, i):
                i += 1
            if i < length:
                n += 1
                while i < length and not PorterStemmer._is_vowel(word, i):
                    i += 1
        return n

    @staticmethod
    def _contains_vowel(word: str) -> bool:
        return any(PorterStemmer._is_vowel(word, i) for i in range(len(word)))

    @staticmethod
    def _ends_double_consonant(word: str) -> bool:
        if len(word) < 2:
            return False
        return word[-1] == word[-2] and not PorterStemmer._is_vowel(word, len(word) - 1)

    @staticmethod
    def _cvc(word: str, i: int) -> bool:
        if i < 2 or not PorterStemmer._is_vowel(word, i - 1) or PorterStemmer._is_vowel(word, i) or PorterStemmer._is_vowel(word, i - 2):
            return False
        return word[i] not in "wxy"

    @classmethod
    def stem(cls, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif not word.endswith("ss") and word.endswith("s"):
            word = word[:-1]

        # Step 1b
        step1b_extra = False
        if word.endswith("eed"):
            stem_part = word[:-3]
            if cls._measure(stem_part) > 0:
                word = word[:-1]
        elif word.endswith("ed"):
            stem_part = word[:-2]
            if cls._contains_vowel(stem_part):
                word = stem_part
                step1b_extra = True
        elif word.endswith("ing"):
            stem_part = word[:-3]
            if cls._contains_vowel(stem_part):
                word = stem_part
                step1b_extra = True

        if step1b_extra:
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif cls._ends_double_consonant(word):
                if word[-1] not in "lsz":
                    word = word[:-1]
            elif cls._measure(word) == 1 and cls._cvc(word, len(word) - 1):
                word += "e"

        # Step 1c
        if word.endswith("y") and cls._contains_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2
        step2_map = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for suffix, repl in step2_map.items():
            if word.endswith(suffix):
                stem_part = word[:-len(suffix)]
                if cls._measure(stem_part) > 0:
                    word = stem_part + repl
                break

        # Step 3
        step3_map = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": ""
        }
        for suffix, repl in step3_map.items():
            if word.endswith(suffix):
                stem_part = word[:-len(suffix)]
                if cls._measure(stem_part) > 0:
                    word = stem_part + repl
                break

        # Step 4
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        ]
        matched = False
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem_part = word[:-len(suffix)]
                if cls._measure(stem_part) > 1:
                    word = stem_part
                matched = True
                break
        if not matched and (word.endswith("sion") or word.endswith("tion")):
            stem_part = word[:-3]
            if cls._measure(stem_part) > 1:
                word = stem_part

        # Step 5a & 5b
        if word.endswith("e"):
            stem_part = word[:-1]
            m = cls._measure(stem_part)
            if m > 1 or (m == 1 and not cls._cvc(stem_part, len(stem_part) - 1)):
                word = stem_part

        if cls._measure(word) > 1 and cls._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]

        return word


def stem_spanish(word: str) -> str:
    """Suffix-stripping stemmer for Spanish matching Rust src/bm25/stemmer.rs."""
    suffixes = (
        "amientos", "imientos", "amiento", "imiento", "izaciones", "ización", "izacion",
        "aciones", "ación", "acion", "amente", "mente", "adoras", "adores", "adora", "ador",
        "idades", "idad", "ancias", "ancia", "encias", "encia",
        "iendo", "ando", "ieron", "aron", "aste", "iste", "emos", "imos",
        "ada", "ado", "ida", "ido", "ar", "er", "ir", "as", "es", "os",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[:-len(suffix)]
    return word


def stem_french(word: str) -> str:
    """Suffix-stripping stemmer for French matching Rust src/bm25/stemmer.rs."""
    suffixes = (
        "issements", "issement", "isations", "isation", "atrices", "ateurs", "atrice", "ateur",
        "ations", "ation", "ements", "ement", "ismes", "isme", "istes", "iste",
        "antes", "ante", "ants", "ant", "euses", "euse", "able", "ible",
        "îtes", "âtes", "rent", "ions", "erie", "era", "ez", "ent", "ais", "ait",
        "er", "ir", "ée", "és", "ées", "é",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[:-len(suffix)]
    return word


def stem_german(word: str) -> str:
    """Suffix-stripping stemmer for German matching Rust src/bm25/stemmer.rs."""
    suffixes = (
        "schaften", "schaft", "heiten", "heit", "keiten", "keit",
        "ungen", "ung", "ieren", "ierend", "ierte", "iert",
        "innen", "isch", "lich", "ende", "end", "ern", "est", "en", "er", "es", "em",
    )
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 3:
            return word[:-len(suffix)]
    return word


def stem_word(word: str, language: str = "english") -> str:
    """Dispatches stemming by language matching Rust src/bm25/stemmer.rs."""
    w = word.lower()
    if len(w) <= 2:
        return w
    lang = language.lower()
    if lang in ("english", "en", "porter"):
        return PorterStemmer.stem(w)
    if lang in ("spanish", "es"):
        return stem_spanish(w)
    if lang in ("french", "fr"):
        return stem_french(w)
    if lang in ("german", "de"):
        return stem_german(w)
    return PorterStemmer.stem(w)

# ---------------------------------------------------------------------------
# Identifier Splitting & CJK Helpers
# ---------------------------------------------------------------------------

_CAMEL_REGEX = re.compile(r"([a-z])([A-Z])")
_ALPHA_DIGIT_REGEX = re.compile(r"([a-zA-Z])([0-9])")
_DIGIT_ALPHA_REGEX = re.compile(r"([0-9])([a-zA-Z])")

def split_identifier(token: str) -> List[str]:
    """Splits camelCase and snake_case tokens into sub-tokens while keeping original."""
    results = [token]
    parts = re.split(r"[_\-]+", token)
    for part in parts:
        if not part:
            continue
        s = _CAMEL_REGEX.sub(r"\1 \2", part)
        s = _ALPHA_DIGIT_REGEX.sub(r"\1 \2", s)
        s = _DIGIT_ALPHA_REGEX.sub(r"\1 \2", s)
        sub_tokens = s.split()
        if len(sub_tokens) > 1 or (len(sub_tokens) == 1 and sub_tokens[0] != token):
            results.extend(sub_tokens)
    return results

def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or
        0x3400 <= cp <= 0x4DBF or
        0x3040 <= cp <= 0x309F or
        0x30A0 <= cp <= 0x30FF or
        0xAC00 <= cp <= 0xD7AF
    )

def cjk_ngrams(text: str) -> List[str]:
    tokens = []
    # Unigrams
    for ch in text:
        tokens.append(ch)
    # Bigrams
    if len(text) >= 2:
        for i in range(len(text) - 1):
            tokens.append(text[i:i+2])
    return tokens

# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalyzerConfig:
    lowercase: bool = True
    stemmer: bool = True
    stopwords: bool = True
    language: str = "english"
    split_identifiers: bool = True
    cjk_bigrams: bool = True


class Analyzer:
    """Multi-lingual lexical analyzer for BM25 hybrid search."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        self.config = config or AnalyzerConfig()
        self._stopwords = get_stopwords(self.config.language) if self.config.stopwords else set()

    def analyze(self, text: str) -> List[str]:
        if not text:
            return []

        # Split into words and CJK chunks
        raw_tokens = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]
            if self.config.cjk_bigrams and is_cjk(ch):
                start = i
                while i < n and is_cjk(text[i]):
                    i += 1
                raw_tokens.extend(cjk_ngrams(text[start:i]))
                continue

            if ch.isalnum() or ch in "_-.$":
                start = i
                while i < n and (text[i].isalnum() or text[i] in "_-.$"):
                    i += 1
                token = text[start:i]
                if self.config.split_identifiers and ("_" in token or "-" in token or any(c.isupper() for c in token)):
                    raw_tokens.extend(split_identifier(token))
                else:
                    raw_tokens.append(token)
                continue

            i += 1

        output = []
        for t in raw_tokens:
            tok = t.lower() if self.config.lowercase else t
            tok = tok.strip("._-")
            if not tok:
                continue

            if self.config.stopwords and tok in self._stopwords:
                continue

            if self.config.stemmer:
                tok = stem_word(tok, self.config.language)

            output.append(tok)

        return output


def tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    """Tokenizes string into terms using default analyzer."""
    config = AnalyzerConfig(
        lowercase=True,
        stopwords=filter_stopwords,
        stemmer=False,
        split_identifiers=False,
        cjk_bigrams=False,
    )
    return Analyzer(config).analyze(text)

