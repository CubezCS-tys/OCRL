"""Rules of the document stage added after the by-eye review of 25 whole documents."""
import numpy as np

from ocrl.doc.parse import Block, PageInfo, TextLine


def L(page, x0, y0, x1, y1, text, region=0, order=0):
    return TextLine(page, region, x0, y0, x1, y1, y1 - 5, 30.0, text, max(len(text) // 2, 1), order)


def test_a_latin_word_broken_at_a_line_end_is_joined_in_the_block_text_only():
    from ocrl.doc.parse import join_lines

    lines = [L(1, 0, 0, 900, 30, "riser ce rayon-"), L(1, 0, 40, 900, 70, "nement et ce"), L(1, 0, 80, 900, 110, "Paris-"),
             L(1, 0, 120, 900, 150, "Match"), L(1, 0, 160, 900, 190, "التربية -"), L(1, 0, 200, 900, 230, "والتعليم")]
    assert Block(lines).text == "riser ce rayonnement et ce Paris- Match التربية - والتعليم"
    assert lines[0].text == "riser ce rayon-"  # the line itself keeps what was read
    assert join_lines(["ab-", "", "cd"]) == "abcd"


def _page(lines, w=1000, h=1400, rules=(), grid=()):
    return PageInfo(1, w, h, lines, [(0, 0, w, h)], list(rules), [], [], list(grid), [], [])


def test_one_row_of_words_or_one_word_is_not_a_table():
    from ocrl.doc.tables import find_tables

    body = [L(1, 100, 100 + 60 * k, 900, 140 + 60 * k, "سطر من النص الجاري في الفقرة") for k in range(4)]
    # a justified line the line finder cut at its spaces, between lines of the paragraph
    row = [L(1, 900 - 150 * k - 120, 340, 900 - 150 * k, 380, "كلمة") for k in range(5)]
    after = [L(1, 100, 400 + 60 * k, 900, 440 + 60 * k, "سطر من النص الجاري في الفقرة") for k in range(3)]
    ls = body + row + after
    kinds = ["paragraph"] * 4 + ["table"] * 5 + ["paragraph"] * 3
    assert find_tables(_page(ls), ls, kinds) == []
    # a single glossary term the model called a table cell
    term = [L(1, 100, 100, 300, 140, "Condom or male sheath")]
    assert find_tables(_page(term + after), term + after, ["table"] + ["paragraph"] * 3) == []
    # three rows of three cells that line up stay a table
    cells = [L(1, 100 + 300 * c, 600 + 60 * r, 300 + 300 * c, 640 + 60 * r, "١٢") for r in range(3) for c in range(3)]
    t = find_tables(_page(body + cells), body + cells, ["paragraph"] * 4 + ["table"] * 9)
    assert len(t) == 1 and (t[0].n_rows, t[0].n_cols) == (3, 3)


def test_lines_of_a_rejected_table_go_back_to_the_paragraph():
    from ocrl.doc.parse import _not_tables

    ls = [L(1, 100, 100, 900, 140, "نص"), L(1, 700, 160, 900, 200, "كلمة"), L(1, 300, 160, 600, 200, "كلمة"),
          L(1, 100, 220, 900, 260, "نص")]
    kinds = ["paragraph", "table", "table", "paragraph"]
    starts = np.array([1.0, 0.9, 0.8, 0.2])
    _not_tables(ls, kinds, starts, {}, [])
    assert kinds == ["paragraph"] * 4 and list(starts[1:3]) == [0.0, 0.0]


def test_a_table_ruled_across_only_is_one_table():
    from ocrl.doc.tables import find_tables

    cap = L(1, 300, 100, 800, 150, "جدول (٣) قائمة الاختبارات")
    head = [L(1, 600, 170, 850, 210, "الذكور أعلى"), L(1, 150, 170, 400, 210, "الاناث أعلى")]
    cells = [L(1, 560 + 20 * (r % 2), 260 + 60 * r, 880, 300 + 60 * r, "اختبار الصور") for r in range(5)]
    cells += [L(1, 120, 260 + 60 * r, 440 - 20 * (r % 2), 300 + 60 * r, "طلاقة لفظية") for r in range(5)]
    # the rules under the header and at the foot, each drawn in two halves
    rules = [(110, 228, 495, 232), (505, 228, 890, 232), (110, 568, 495, 572), (505, 568, 890, 572)]
    ls = [cap] + head + cells
    kinds = ["table_caption", "paragraph", "noise"] + ["paragraph", "table"] * 5
    t = find_tables(_page(ls, rules=rules), ls, kinds)
    assert len(t) == 1 and t[0].n_cols == 2 and len(t[0].lines) == 12


def test_a_table_foot_drawn_in_two_halves_is_not_the_rule_over_footnotes():
    from ocrl.doc.parse import _footnotes_under_rule

    text = [L(1, 100, 100 + 50 * k, 900, 130 + 50 * k, "نص") for k in range(8)]
    under = L(1, 100, 900, 900, 930, "نص تحت الجدول")
    halves = [(100, 860, 495, 864), (505, 860, 900, 864)]
    p = _page(text + [under], rules=halves)
    b = Block([under])
    _footnotes_under_rule([p], [b])
    assert b.type == "paragraph"
    p = _page(text + [under], rules=halves[1:])  # one short rule at the column's edge: the notes' rule
    _footnotes_under_rule([p], [b])
    assert b.type == "footnote"


def test_a_running_head_with_text_above_it_is_inside_the_page():
    from ocrl.doc.parse import _heads_in_text

    ls = [L(1, 600, 40, 900, 70, "مجلة العلوم"), L(1, 500, 150, 900, 190, "نص الفقرة الأولى هنا"), L(1, 500, 200, 900, 240, "ونص الفقرة يستمر هنا"),
          L(1, 100, 300, 300, 340, "المجتمع", region=1), L(1, 100, 350, 300, 600, "", region=1)]
    kinds = ["running_head", "paragraph", "paragraph", "running_head", "noise"]
    _heads_in_text(ls, kinds)
    assert kinds == ["running_head", "paragraph", "paragraph", "figure", "noise"]


def test_a_big_running_head_on_the_first_page_is_the_title():
    from ocrl.doc.parse import _first_page_title

    p1, p2 = _page([]), _page([])
    p2.number = 2
    head1 = Block([L(1, 300, 40, 700, 70, "اثر تطبيق قاعدة كومسوال")], "running_head")
    title = Block([L(1, 100, 300, 900, 360, "اثر تطبيق قاعدة كومسوال في موازنة")], "running_head")
    head2 = Block([L(2, 300, 40, 700, 70, "اثر تطبيق قاعدة كومسوال")], "running_head")
    _first_page_title([p1, p2], [head1, title, head2])
    assert [b.type for b in (head1, title, head2)] == ["running_head", "title", "running_head"]


def test_a_poem_in_two_half_lines_is_read_across():
    from ocrl.doc.parse import _couplets

    right = [L(1, 520, 100 + 60 * k, 900, 140 + 60 * k, f"صدر {k}", region=0, order=k) for k in range(5)]
    left = [L(1, 100, 100 + 60 * k, 480, 140 + 60 * k, f"عجز {k}", region=1, order=k) for k in range(5)]
    ls, kinds, starts, paired = _couplets(right + left, ["verse"] * 10, np.zeros(10))
    assert [l.text for l in ls[:4]] == ["صدر 0", "عجز 0", "صدر 1", "عجز 1"]
    assert set(kinds) == {"verse"} and len(paired) == 10
    # two columns of prose side by side are left alone
    ls2, *_ = _couplets(right + left, ["paragraph"] * 10, np.zeros(10))
    assert [l.text for l in ls2[:2]] == ["صدر 0", "صدر 1"]


def test_pieces_of_a_formula_are_one_block():
    from ocrl.doc.parse import _join_formulas

    p = _page([L(1, 100, 100, 900, 140, "نص")] * 3)
    pieces = [Block([L(1, 400, 500, 500, 530, "AT")], "equation"), Block([L(1, 330, 520, 390, 560, "Cf =")], "equation"),
              Block([L(1, 0, 0, 2, 2, "")], "noise"), Block([L(1, 410, 545, 480, 580, "OR")], "equation"),
              Block([L(1, 800, 520, 880, 560, "(1)")], "equation"), Block([L(1, 100, 700, 900, 740, "نص")], "paragraph")]
    _join_formulas([p], pieces)
    assert [b.type for b in pieces] == ["equation", "noise", "paragraph"]
    assert pieces[0].text.startswith("Cf =")


def test_a_title_on_two_lines_is_one_block_but_its_translation_is_not():
    from ocrl.doc.parse import _join_titles

    blocks = [Block([L(1, 200, 100, 800, 150, "دراسة لاهم العوامل")], "title"), Block([L(1, 250, 160, 750, 210, "المؤثرة على وزن الطفل")], "title"),
              Block([L(1, 200, 230, 800, 280, "A study of the factors")], "title")]
    _join_titles(blocks)
    assert [len(b.lines) for b in blocks] == [2, 1]
