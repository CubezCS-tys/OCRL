from ocrl.pipeline import Cluster, _line_text


def line(*words):
    """Each word is given as the clusters read off the page, right to left."""
    out = []
    for w in words:
        cl = [Cluster(0, [0], t, 0.0) for t in w]
        cl[-1].space_after = True
        out.extend(cl)
    return _line_text(out)


def test_arabic_is_already_in_reading_order():
    assert line(["كتا", "ب"], ["محمد"]) == "كتاب محمد"


def test_numbers_are_turned_around():
    assert line(["عا", "م"], ["2", "8", "9", "1"]) == "عام 1982"


def test_a_latin_phrase_is_turned_around_as_a_whole():
    # On the page, right to left: Arabic word, then "Theory" (met first, from its end), then "Power".
    assert line(["قو", "ة"], list("yroehT"), list("rewoP"), ["في"]) == "قوة Power Theory في"
