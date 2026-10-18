"""
preprocessing.py
----------------
Text-cleaning helpers shared by train_model.py and app.py.

Keeping this logic in ONE place guarantees that the text is cleaned in
exactly the same way during training and during prediction.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Words such as "the", "is", "and" carry little meaning for spam detection.
# We keep a few words that are often useful signals in spam ("free", "win"
# are not in the list anyway), and we remove very common filler words.
STOP_WORDS = set(ENGLISH_STOP_WORDS)

# Pre-compiled regular expressions (faster than compiling on every call).
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
NUMBER_PATTERN = re.compile(r"\d+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a raw message and return a normalised string.

    Steps:
        1. Convert to lowercase.
        2. Replace URLs with the token 'urltoken' (links are a strong spam signal).
        3. Replace numbers with the token 'numtoken' (phone numbers, prizes, ...).
        4. Remove punctuation and special characters.
        5. Remove stop words.
        6. Remove extra whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                              # 1. lowercase
    text = URL_PATTERN.sub(" urltoken ", text)       # 2. URLs
    text = NUMBER_PATTERN.sub(" numtoken ", text)    # 3. numbers
    text = NON_LETTER_PATTERN.sub(" ", text)         # 4. punctuation / symbols

    # 5. remove stop words (and single-letter leftovers such as "s" from "it's")
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]

    # 6. join with single spaces (this also removes extra whitespace)
    return WHITESPACE_PATTERN.sub(" ", " ".join(words)).strip()
