"""Phrase matcher used by the retrieval evaluation (must_contain checks).

Both the chunk text and the required phrase are normalised: lower case, whitespace collapsed, soft hyphens
dropped, and "-", "/" and space treated as equivalent (so "tipping off" matches "Tipping-off", and
"Tumbler/Mixer" matches "tumbler mixer"). Hyphenation at line breaks is handled by also trying the text with
"word- word" joined into "wordword". A must_contain entry is one or more alternative phrases separated by "|";
it is a literal phrase, not a regular expression.
"""
import re

_SOFT_HYPHEN = "­"


def normalize(text):
    text = text.replace(_SOFT_HYPHEN, "").lower()
    return re.sub(r"[-‐‑–—/\s]+", " ", text).strip()


def _variants(text):
    text = text.replace(_SOFT_HYPHEN, "")
    # a hyphen followed by a line break or space between lowercase letters may be a hyphenated word: join it
    joined = re.sub(r"(?<=[A-Za-z])[-‐]\s+(?=[a-z])", "", text)
    return {normalize(text), normalize(joined)}


def contains_phrase(text, phrase):
    """True if any '|'-separated alternative of `phrase` occurs in `text` after normalisation."""
    variants = _variants(text)
    for alt in phrase.split("|"):
        alt = normalize(alt)
        if not alt:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(alt) + r"(?![a-z0-9])"  # whole words only
        if any(re.search(pattern, v) for v in variants):
            return True
    return False


def matches_all(text, phrases):
    return all(contains_phrase(text, p) for p in phrases)
