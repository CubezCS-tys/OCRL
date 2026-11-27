from ocrl.structure.tag import _ITEM, _mark_key, _steps


def test_item_marks():
    for t in ["أ-الشبكات الاجتماعية", "ب- خدمة الويب", "١. بعد الانتهاء", "(3) نص", "- نص", "• نقطة"]:
        assert _ITEM.match(t), t
    for t in ["أخبرت عن خليفة", "لـ 5 سنوات", "2010) قال"]:
        assert not _ITEM.match(t), t


def test_marks_that_step_by_one():
    one, two, alif, ba, jim = (_mark_key(t) for t in ["١. بعد", "٢. بعد", "أ-الشبكات", "ب-خدمة", "ج-خدمة"])
    assert _steps(one, two) and not _steps(two, one)
    assert _steps(alif, ba) and _steps(ba, jim)  # in either order of the alphabet
    assert not _steps(one, _mark_key("2) نص"))  # another kind of mark
    assert _mark_key("- نص") is None  # a dash has no place in a sequence


def test_page_furniture_needs_two_pages():
    from ocrl.doc.parse import Block, PageInfo, TextLine, _page_furniture

    def page(n, folio):
        top = TextLine(n, 0, 100, 40, 900, 70, 65, 30, "مجلة الدراسات التربوية العدد الثالث", 30)
        body = TextLine(n, 0, 100, 400, 900, 430, 425, 30, "نص الصفحة", 10)
        foot = TextLine(n, 0, 480, 1450, 520, 1480, 1475, 30, folio, 3)
        return PageInfo(n, 1000, 1500, [top, body, foot], [(0, 0, 1000, 1500)], [], []), [Block([top]), Block([body]), Block([foot], "noise")]

    (p1, b1), (p2, b2) = page(4, "١٢"), page(5, "١٣")
    _page_furniture([p1, p2], b1 + b2)
    assert [b.type for b in b1 + b2] == ["running_head", "paragraph", "page_number"] * 2
    (p1, b1), (p2, b2) = page(4, "١٢"), page(5, "٤٠")  # numbers that do not step with the pages
    _page_furniture([p1, p2], b1 + b2)
    assert b1[2].type == b2[2].type == "noise"


def test_margin_lines_need_same_words_same_height():
    from ocrl.doc.parse import PageInfo, TextLine, _margin_lines

    def page(n, head_y, note):
        head = TextLine(n, 0, 300, head_y, 700, head_y + 30, head_y + 25, 30, "وجهة نظرهم", 10)
        body = TextLine(n, 0, 100, 400, 900, 430, 425, 30, "نص الصفحة", 10)
        foot = TextLine(n, 0, 100, note, 900, note + 30, note + 25, 30, "(٣) المرجع السابق، ص ١٢", 20)
        return PageInfo(n, 1000, 1500, [head, body, foot], [(0, 0, 1000, 1500)], [], [])

    found = _margin_lines([page(4, 100, 1360), page(5, 102, 1420)])
    assert found == [{0: "running_head"}, {0: "running_head"}]  # a note that comes back is not a running foot
    assert _margin_lines([page(4, 100, 1360), page(5, 160, 1420)]) == [{}, {}]  # not at the same height
    assert _margin_lines([page(4, 100, 1360)]) == [{}]  # one page shows nothing


def test_paragraph_ends_at_short_line_that_closes_a_sentence():
    import numpy as np
    from ocrl.structure.tag import _paragraph_ends

    # Arabic: lines run from the right edge (x1 = 1000); the third stops short and ends with a full stop
    boxes = np.array([[100, y, 1000, y + 30] for y in range(0, 400, 50)], np.float32)
    boxes[2, 0] = 600
    texts = ["سطر كامل"] * 8
    texts[2] = "آخر الفقرة."
    out = _paragraph_ends(np.zeros(8), texts, boxes)
    assert out[3] >= 0.5 and (out[[1, 2, 4, 5]] < 0.5).all()
    texts[2] = "آخر الفقرة"  # no full stop: not sure, left to the model
    assert (_paragraph_ends(np.zeros(8), texts, boxes) < 0.5).all()
