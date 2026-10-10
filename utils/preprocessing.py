# utils/preprocessing.py

import re
import pkg_resources

from symspellpy.symspellpy import SymSpell, Verbosity

from .constants import abbreviations


# --- SymSpell Setup ---
sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7,
)

dictionary_path = pkg_resources.resource_filename(
    "symspellpy",
    "frequency_dictionary_en_82_765.txt",
)

sym_spell.load_dictionary(
    dictionary_path,
    term_index=0,
    count_index=1,
)


# --- Text Normalization ---
def normalize_text(text):
    """Normalize text while preserving course codes and hyphens."""
    text = str(text or "")

    # Replace punctuation with spaces, but retain hyphens.
    text = re.sub(r"[^a-zA-Z0-9\s-]", " ", text)

    # Reduce repeated letters: e.g. 'hellooo' -> 'helo'.
    text = re.sub(r"([a-zA-Z])\1{2,}", r"\1", text)

    # Remove unnecessary spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --- Abbreviation Expansion ---
def expand_abbreviations(text):
    """Expand known abbreviations without case-sensitive key problems."""
    result = text

    # Handle multiword abbreviation keys first.
    multiword = sorted(
        [
            (key, value)
            for key, value in abbreviations.items()
            if " " in str(key).strip()
        ],
        key=lambda item: len(str(item[0])),
        reverse=True,
    )

    for key, value in multiword:
        pattern = r"(?<!\w)" + re.escape(str(key)) + r"(?!\w)"
        result = re.sub(
            pattern,
            str(value),
            result,
            flags=re.IGNORECASE,
        )

    # Handle single-word abbreviations.
    singleword = {
        str(key).lower(): str(value)
        for key, value in abbreviations.items()
        if " " not in str(key).strip()
    }

    expanded_words = []

    for word in result.split():
        replacement = singleword.get(word.lower(), word)
        expanded_words.append(replacement)

    return " ".join(expanded_words)


# --- Main Preprocessing ---
def preprocess_text(text):
    """Normalize text and correct ordinary words without altering codes."""
    text = normalize_text(text)
    text = expand_abbreviations(text)

    corrected_words = []

    for word in text.split():
        # Preserve course codes, numeric terms, hyphenated codes,
        # and uppercase acronyms instead of spell-correcting them.
        contains_digit = any(char.isdigit() for char in word)
        contains_hyphen = "-" in word
        is_acronym = word.isupper() and any(
            char.isalpha() for char in word
        )

        if (
            contains_digit
            or contains_hyphen
            or is_acronym
            or len(word) < 4
            or not word.isalpha()
        ):
            corrected_words.append(word)
            continue

        # Avoid changing capitalized names and terms.
        if word[0].isupper():
            corrected_words.append(word)
            continue

        suggestions = sym_spell.lookup(
            word.lower(),
            Verbosity.CLOSEST,
            max_edit_distance=2,
        )

        if suggestions:
            corrected_words.append(suggestions[0].term)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


# --- Course Code Prefix Extraction ---
def extract_prefix(code):
    """Extract the alphabetic prefix from a course code."""
    match = re.match(r"([A-Za-z]+(?:-[A-Za-z]+)*)", str(code).strip())

    return match.group(1).upper() if match else None