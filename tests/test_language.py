"""Language detection, with special care that ordinary English is never tagged Hinglish."""

from __future__ import annotations

import pytest

from nlp.language import ANALYSED, COMMON_ENGLISH, HINDI_WORDS, detect_language, is_hinglish


@pytest.mark.parametrize(
    "text",
    [
        "The main issue is the price, it is way too high for what you get.",
        "Hai everyone, the main problem with this serum is that it pilled under my sunscreen.",  # 'hai' as a typo of hi
        "I bought it at the main market in Mumbai and the main shop had it in stock.",
        "To be honest the pencil is fine, but the main thing is the lead breaks so easily.",
        "He said it is so good, and he is right, the eraser does not smudge at all.",
        "Main character energy. This toner is amazing for my skin, ka-ching!",
    ],
)
def test_english_with_hindi_lookalike_words_is_never_hinglish(text: str) -> None:
    assert not is_hinglish(text)
    assert detect_language(text) == "en"


@pytest.mark.parametrize(
    "text",
    [
        "Ye pencil bahut accha hai, smudge nahi karta",
        "Bhai ye sharpener bekar hai, lead tut jati hai baar baar",
        "Mujhe ye cream bahut pasand aayi, skin pe koi irritation nahi hua",
        "Price thoda zyada hai lekin quality kaafi acchi hai",
    ],
)
def test_hinglish(text: str) -> None:
    assert is_hinglish(text)
    assert detect_language(text) == "hinglish"


def test_hinglish_needs_three_distinct_words_and_a_share() -> None:
    assert not is_hinglish("hai hai hai this is the best pen I have ever used in my whole life honestly")  # 1 distinct word
    assert not is_hinglish("ye bahut accha " + "this pen writes smoothly and the ink dries fast " * 3)  # 3 words, low share


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("이 크림 정말 좋아요 촉촉하고 자극이 없어요", "ko"),
        ("यह पेंसिल बहुत अच्छी है और आसानी से नहीं टूटती", "hi"),
        ("Une perte de temps, le produit ne fonctionne pas du tout", "fr"),
        ("This moisturiser broke me out within two days, really disappointed", "en"),
        ("Me gustaría que al realizar una búsqueda en la aplicación te dirija a la nota", "es"),
        ("Ótimo aplicativo com diversas ferramentas e blocos gratuitos e de qualidade", "pt"),
    ],
)
def test_other_languages(text: str, code: str) -> None:
    assert detect_language(text) == code


@pytest.mark.parametrize("text", ["Sync broke", "Love it!", "", "👍👍", "12345"])
def test_short_or_empty_is_undetermined_and_still_analysed(text: str) -> None:
    assert detect_language(text) == "und"
    assert "und" in ANALYSED and "en" in ANALYSED and "hinglish" not in ANALYSED


def test_word_list_has_no_everyday_english() -> None:
    assert not HINDI_WORDS & COMMON_ENGLISH
    for word in ("main", "to", "the", "is", "me", "he", "hi", "so", "do", "in"):
        assert word not in HINDI_WORDS



@pytest.mark.parametrize(
    "text",
    [  # real mentions that lingua's low-accuracy mode tagged fr / it / nl / pt
        "The best application for professional work",
        "my favorite app of all times!",
        "This is the absolute best word processor ever. Love it.",
        "My Neovim setup on termux have better performance than ts.",
    ],
)
def test_short_english_is_not_mistaken_for_other_languages(text: str) -> None:
    assert detect_language(text) == "en"


def test_links_are_ignored() -> None:
    assert detect_language("Simplified Obsidian https://fed.brid.gy/r/https://rastundrost.de/2026/10/12/simplified-obsidian") == "und"


@pytest.mark.parametrize(
    "text",
    [  # real YouTube comments that lingua tagged nl / de with confidence up to 0.94
        "doms offer better in same price",
        "Best gel exam pen under 50rs",
        "50 pages so thats 25 sheet",
        "Wow remote control car",
    ],
)
def test_short_latin_text_defaults_to_english(text: str) -> None:
    assert detect_language(text) == "en"


@pytest.mark.parametrize(
    ("text", "code"),
    [  # short, but lingua is >= 0.95 sure: still not English
        ("Me encanta este producto", "es"),
        ("Das ist sehr gut", "de"),
        ("Ik vind dit product geweldig", "nl"),
        ("Muito bom, recomendo demais", "pt"),
        ("Sangat cocok untuk kulit berminyak", "id"),
    ],
)
def test_confident_short_foreign_text_keeps_its_language(text: str, code: str) -> None:
    assert detect_language(text) == code


@pytest.mark.parametrize("text", ["ကောင်း", "אחת האפליקציות המעולות ביותר לניהול מידע"])
def test_short_text_rule_is_latin_only(text: str) -> None:  # scripts without a SCRIPTS entry stay undetermined
    assert detect_language(text) == "und"
