"""Language of a mention: ISO code ('en', 'hi', 'ko', …), 'hinglish' or 'und' (undetermined).

The NLP models are English-only, so only 'en' and 'und' are analysed; other languages are stored
and reported, and wait for translation (Step 10).

- Scripts decide most languages exactly (Hangul -> ko, Bengali -> bn, …). lingua only chooses within
  Latin and Devanagari, loaded with just those few languages in low-accuracy mode to keep memory low
  on an 8 GB Mac (its low-accuracy mode alone called Devanagari Hindi "Bengali").
- Romanised Hindi ("Hinglish") is detected by word list, since detectors call it English or
  Indonesian. Words that are also common English words ("main", "to", "the", "is", "me", "he",
  "hi", "so", "do", "in", "ha") are not on the list, and a text needs at least MIN_HINDI_WORDS
  distinct Hindi words AND at least MIN_HINDI_SHARE of its words, so normal English reviews that
  happen to say "main" or "hai" are never tagged Hinglish.
- Latin text with a couple of unambiguous English words ("the", "my", "of", "than"…) is English before
  lingua is asked: its low-accuracy mode called short English French, Italian or Dutch.
- Very short or low-confidence texts are 'und' rather than guessed ("Sync broke" looks Dutch).
- Short Latin texts (under SHORT_WORDS words) are English unless lingua is at least HIGH_CONFIDENCE sure of
  another language. Measured on real YouTube comments: misread English scored up to 0.94 ("Best gel exam
  pen under 50rs" -> nl 0.94, "doms offer better in same price" -> nl 0.81), real short foreign sentences
  0.96-1.00 ("Me encanta este producto" es 0.96, "Das ist sehr gut" de 1.00). The margin is thin.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

ANALYSED = ("en", "und")  # languages the English models handle (und = too short or unclear to tell)
MIN_WORDS = 4
MIN_CONFIDENCE = 0.5
SHORT_WORDS = 8  # Latin texts with fewer words default to English...
HIGH_CONFIDENCE = 0.95  # ...unless lingua is this sure of another language
MIN_HINDI_WORDS = 3
MIN_HINDI_SHARE = 0.2

# Romanised Hindi function and common words, excluding anything that is also a common English word.
HINDI_WORDS = frozenset("""
hai hain nahi nahin nhi bahut bohot bahot accha acha achha achchha kya kyu kyun kyon mein mujhe mujhko
bhi aur ka ki ke ko se ye yeh woh wo vo lekin kuch sab wala wali wale raha rahi rahe karta karti karte
karna kiya kiye diya diye liya liye tha thi the_ hota hoti hote bhai yaar sahi bekar bekaar paisa paise kaafi
zyada jyada jada sasta saste mehenga mehnga chalta chalti hum aap tum apna apni isme isko usko kaise kab
abhi bilkul thoda thodi bas matlab samajh pata dekho dekha lagta lagti accha hi_ jaisa jaise waise bohut
""".split()) - {"the_", "hi_"}
# Unambiguous English function words (not also common in Spanish/Portuguese/Italian/French/Dutch, so
# "a", "me", "no", "do", "de", "en" are left out). lingua's low-accuracy mode tags short English as
# French/Italian/Dutch ("my favorite app of all times!" -> it), so these decide first.
ENGLISH_MARKERS = frozenset("""
the and is are was were this that these those it its my your our their you they we he she of for with
have has had but not very all than just been what when which would could should will because about
""".split())
MIN_ENGLISH_MARKERS = 2
MIN_ENGLISH_SHARE = 0.15
COMMON_ENGLISH = frozenset("main to the is me he hi so do in ha a i it on at no not be by an as or".split())
assert not HINDI_WORDS & COMMON_ENGLISH  # the list must never contain everyday English words

_LATIN = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z']+|[^\W\d_]+", re.UNICODE)


# Scripts decide most languages exactly; lingua only chooses within a script (Latin, Devanagari).
SCRIPTS = (  # (first, last code point, language or None = ask lingua)
    (0xAC00, 0xD7AF, "ko"), (0x1100, 0x11FF, "ko"), (0x3130, 0x318F, "ko"),  # Hangul
    (0x3040, 0x30FF, "ja"),  # hiragana / katakana
    (0x4E00, 0x9FFF, "zh"),  # Han (also used by Japanese; kana above wins when present)
    (0x0980, 0x09FF, "bn"), (0x0B80, 0x0BFF, "ta"), (0x0C00, 0x0C7F, "te"), (0x0E00, 0x0E7F, "th"),
    (0x0600, 0x06FF, "ur"),  # Arabic script (Urdu for our audience)
    (0x0900, 0x097F, None),  # Devanagari: Hindi or Marathi
)


def _script_counts(text: str) -> dict[str | None, int]:
    counts: dict[str | None, int] = {}
    for ch in text:
        code = ord(ch)
        for first, last, lang in SCRIPTS:
            if first <= code <= last:
                key = lang if lang is not None else "devanagari"
                counts[key] = counts.get(key, 0) + 1
                break
    return counts


@lru_cache(maxsize=2)
def _detector(script: str) -> Any:
    """Low-accuracy lingua with only the languages that share the script (small and fast)."""
    from lingua import Language, LanguageDetectorBuilder

    if script == "devanagari":
        languages = [Language.HINDI, Language.MARATHI]
    else:
        languages = [Language.ENGLISH, Language.INDONESIAN, Language.SPANISH, Language.PORTUGUESE, Language.FRENCH,
                     Language.GERMAN, Language.ITALIAN, Language.DUTCH, Language.VIETNAMESE]
    return LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()


def hindi_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w in HINDI_WORDS}


def is_hinglish(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return False
    found = hindi_words(text)
    share = sum(w in HINDI_WORDS for w in words) / len(words)
    return len(found) >= MIN_HINDI_WORDS and share >= MIN_HINDI_SHARE


def looks_english(words: list[str]) -> bool:
    lowered = [w.lower() for w in words]
    hits = [w for w in lowered if w in ENGLISH_MARKERS]
    return len(hits) >= MIN_ENGLISH_MARKERS and len(hits) / len(lowered) >= MIN_ENGLISH_SHARE


def detect_language(text: str | None) -> str:
    """'en', another ISO 639-1 code, 'hinglish', or 'und'."""
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text or "").strip()  # links say nothing about language
    words = _WORD.findall(text)
    if not words:
        return "und"
    latin = len(_LATIN.findall(text)) >= 0.5 * sum(len(w) for w in words)
    if latin and is_hinglish(text):
        return "hinglish"
    if latin and looks_english(words):
        return "en"
    if latin and len(words) < MIN_WORDS:
        return "und"
    scripts = _script_counts(text)
    if scripts and not latin:
        script = max(scripts, key=scripts.__getitem__)
        if script == "ja" or (script == "zh" and "ja" in scripts):
            return "ja"
        if script != "devanagari":
            return script
        best = _detector("devanagari").detect_language_of(text)
        return best.iso_code_639_1.name.lower() if best else "hi"
    confidences = _detector("latin").compute_language_confidence_values(text)
    if not confidences:
        return "und"
    best = confidences[0]
    code = best.language.iso_code_639_1.name.lower()
    if latin and len(words) < SHORT_WORDS and code != "en" and best.value < HIGH_CONFIDENCE:
        return "en"
    if best.value < MIN_CONFIDENCE:
        return "und"
    return code
