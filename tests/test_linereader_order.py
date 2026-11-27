"""Reading order of a line reader's output: the bidi layout and its inverse (ocrl.recog.linereader)."""
import torch

from ocrl.recog.linereader import (LineNet, LineReader, bidi_levels, logical_to_reader, old_rule_to_logical,
                                   reader_to_logical, to_logical)

# Logical text -> what a perfect reader of the mirrored line emits (right to left on the paper),
# worked out by hand from UAX #9.
CASES = [
    # European digits after an Arabic letter are Arabic numbers (W2); a hyphen between them is not
    # joined (W4 joins ES only between European numbers), so the date is three runs.
    ("في 2018-02-23 م", "في 8102-20-32 م"),
    # a slash (CS) between Arabic numbers joins them: one run
    ("في 2018/02/23 م", "في 32/20/8102 م"),
    # Arabic-Indic digits with the Arabic decimal sign: one run
    ("المتوسط ٢٠٫٥٧ فقط", "المتوسط ٧٥٫٠٢ فقط"),
    # the percent sign stays outside the number in Arabic text: printed "%45"
    ("بنسبة 45% من", "بنسبة 54% من"),
    # a citation: the Latin phrase with its year is one left-to-right run; the brackets around it are
    # right-to-left (they hold nothing right-to-left but the text before them is Arabic)
    ("قال (Smith, 2006) إن", "قال (6002 ,htimS) إن"),
    # brackets inside a Latin run come out mirrored, since the image is flipped
    ("انظر Bernstein (2005) أيضا", "انظر (5002) nietsnreB أيضا"),
    ("عند p < 0.01 فقط", "عند 10.0 > p فقط"),
    # two Latin names separated by Arabic: each its own run, in right-to-left order
    ("دراسة Smith و Jones معا", "دراسة htimS و senoJ معا"),
    # Arabic-Indic numbers in a citation
    ("(أندرسون، ٢٠٠٦، ١٢١)", "(أندرسون، ٦٠٠٢، ١٢١)"),
]


def test_forward_layout():
    for logical, emitted in CASES:
        assert logical_to_reader("ا" + logical)[1:] == emitted, logical


def test_inverse_restores_reading_order():
    for logical, emitted in CASES:
        assert reader_to_logical(emitted, mirror=True, prefer="phrase", ranges_low_high=False) == logical, emitted


def test_left_to_right_line_is_turned_round():
    assert reader_to_logical("(9102) .p ,htimS", mirror=True) == "Smith, p. (2019)"
    # v7 writes brackets as they stand in reading order (measured on real lines), so no mirroring
    assert reader_to_logical(")9102( .p ,htimS") == "Smith, p. (2019)"
    assert reader_to_logical("51 - 01") == "10 - 15"  # a table cell of numbers only reads left to right


def test_v7_defaults():
    # each left-to-right piece separate, in right-to-left order, as Azure writes it (same layout on paper)
    assert reader_to_logical("قال (6002 ,htimS) إن") == "قال (2006 ,Smith) إن"
    assert reader_to_logical("تاريخ الإرسال: 32-20-8102") == "تاريخ الإرسال: 23-02-2018"
    assert reader_to_logical("تاريخ الإرسال: 8102-20-32") == "تاريخ الإرسال: 23-02-2018"  # printed as one run
    # a two-number range is written low to high, whichever way it was printed
    assert reader_to_logical("بين (21-6) سنة") == "بين (6-12) سنة"
    assert reader_to_logical("بين (6-21) سنة") == "بين (6-12) سنة"
    assert reader_to_logical("المتوسط ٧٥٫٠٢ فقط") == "المتوسط ٢٠٫٥٧ فقط"


def test_levels():
    #           ق ا ل ␣ ( S m i t h ,  ␣ 2 0 0 6 ) ␣ إ ن
    assert bidi_levels("قال (Smith, 2006) إن") == [1, 1, 1, 1, 1] + [2] * 11 + [1, 1, 1, 1]
    assert bidi_levels("abc", 0) == [0, 0, 0]


def test_old_rule_kept_unchanged():
    # what our own readers were trained with: its own inverse, and the date bug it has
    assert old_rule_to_logical is to_logical
    assert to_logical("في 8102-20-32 م") == "في 23-02-2018 م"
    for t in ["قال 2006 Smith إن", "Hello 12", "في ٢٠٫٥٧ و 45.5 م"]:
        assert to_logical(to_logical(t)) == t


def _ckpt(tmp_path, meta_extra):
    chars = list("اب12")
    net = LineNet(len(chars) + 1, hid=4, feat=8, base=2)
    p = tmp_path / "r.pt"
    torch.save({"model": net.state_dict(), "meta": {"chars": chars, "H": 48, "hid": 4, "feat": 8, **meta_extra}}, p)
    return str(p)


def test_rule_is_chosen_per_reader(tmp_path):
    assert LineReader(_ckpt(tmp_path, {}), device="cpu").order == "bidi"  # the owner's readers
    ours = _ckpt(tmp_path, {"made_by": "OCRL scripts/train_line_reader.py"})
    r = LineReader(ours, device="cpu")
    assert r.order == "old" and r.to_reading_order("المتوسط ٧٥٫٠٢ فقط") == "المتوسط ٥٧٫٢٠ فقط"
    assert LineReader(ours, device="cpu", order="bidi").to_reading_order("المتوسط ٧٥٫٠٢ فقط") == "المتوسط ٢٠٫٥٧ فقط"
