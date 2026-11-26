from ocrl.rules.joining import (
    first_violation,
    forces_word_break,
    is_valid_unit,
    joining_type,
    joins,
    positional_forms,
    text_units,
    units,
)
from ocrl.rules.signature import satellite_range
from ocrl.text import clean_sentence, runs


def test_joining_types_come_from_unicode():
    assert joining_type("ب") == "D"
    assert all(joining_type(c) == "R" for c in "ادذرزوةأإآؤ")
    assert joining_type("ى") == "D"  # Unicode's call; in Arabic it is word-final anyway
    assert joining_type("ء") == "U"
    assert joining_type("َ") == "T"  # fatha
    assert joining_type("ـ") == "C"
    assert joining_type("5") == "U"


def test_joins():
    assert joins("ب", "ا") and joins("ب", "ب")
    assert not joins("ا", "ب") and not joins("ب", "ء") and not joins("ء", "ا")


def test_units_are_subwords():
    assert units("كتاب") == ["كتا", "ب"]
    assert units("مدرسة") == ["مد", "ر", "سة"]
    assert units("شيء") == ["شي", "ء"]
    assert units("دار") == ["د", "ا", "ر"]
    assert units("محمد") == ["محمد"]
    assert units("الأرض") == ["ا", "لأ", "ر", "ض"]


def test_marks_stay_with_their_letter():
    assert units("كِتَابٌ") == ["كِتَا", "بٌ"]


def test_digits_and_punctuation_are_single_units():
    assert units("1973م)") == ["1", "9", "7", "3", "م", ")"]
    assert text_units("في عام 12") == ["في", "عا", "م", "1", "2"]


def test_valid_unit():
    assert is_valid_unit("كتا") and is_valid_unit("ب") and is_valid_unit("ء")
    assert not is_valid_unit("كتاب")  # alef cannot be mid-body
    assert first_violation("كتاب") == 3
    assert first_violation("محمد") is None


def test_word_break_is_forced_only_when_letters_would_have_joined():
    assert forces_word_break("كتب", "محمد")  # ب would join م: must be two words
    assert forces_word_break("من", "ا")
    assert not forces_word_break("كتا", "ب")  # after alef: no information
    assert not forces_word_break("شي", "ء")  # hamza never joins: no information
    assert not forces_word_break("عا", "م")


def test_positional_forms():
    assert positional_forms("كتاب") == ["initial", "medial", "final", "isolated"]
    assert positional_forms("بيت") == ["initial", "medial", "final"]


def test_satellite_range():
    assert satellite_range("كتا") == (1, 2)  # ت: one or two dot blobs
    assert satellite_range("لأ") == (1, 1)
    assert satellite_range("محمد") == (0, 0)
    assert satellite_range("ك") == (0, 1)
    assert satellite_range("شي") == (2, 5)


def test_runs_and_cleaning():
    assert runs("و1000") == [("ar", "و"), ("num", "1000")]
    assert runs("(1973م)") == [("p", "("), ("num", "1973"), ("ar", "م"), ("p", ")")]
    assert runs("abc") == [("lat", "abc")]
    assert runs("ab\u00a0cd") == [("lat", "ab\u00a0cd")]
    assert runs("ab~") == []
    assert clean_sentence("كَتَبَ الولد té الدرس.") == "كتب الولد الدرس."


def test_vowel_marks_are_ink_to_account_for():
    assert satellite_range("كَتَا") == (3, 4)  # two marks, plus the dots of ta
    assert satellite_range("مّ") == (1, 1)
    assert satellite_range("مَّ") == (1, 2)  # shadda and fatha may fuse


def test_latin_phrases_stay_together():
    from ocrl.text import NBSP, join_latin
    assert join_latin(["نظرية", "Power", "Theory", "في"]) == ["نظرية", "Power" + NBSP + "Theory", "في"]
