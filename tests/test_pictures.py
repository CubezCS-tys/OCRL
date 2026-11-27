"""Equations and pictures shown as crops of the scan: one figure block per picture, its words its labels."""
from ocrl.doc.parse import Block, PageInfo, TextLine, _picture_members, shows_words


def L(x0, y0, x1, y1, text):
    return TextLine(1, 0, x0, y0, x1, y1, y1 - 5, 30.0, text, max(len(text) // 2, 1), 0)


def _page(lines, pictures=(), drawings=()):
    return PageInfo(1, 1000, 1400, lines, [(0, 0, 1000, 1400)], [], [], [], [], [], [], [], list(pictures), 0, list(drawings))


def test_lines_inside_a_drawing_are_its_labels_and_a_photograph_with_none_stands_where_it_is():
    above = L(100, 100, 900, 140, "سطر من النص الجاري فوق الخريطة في الصفحة")
    labels = [L(300, 400, 380, 430, "العراق"), L(600, 500, 680, 530, "ايران")]
    caption = L(300, 820, 700, 860, "شكل 1: خريطة المملكة")
    below = L(100, 1000, 900, 1040, "سطر من النص الجاري تحت الصورة في الصفحة")
    ls = [above] + labels + [caption, below]
    kinds = ["paragraph", "table", "figure", "figure_caption", "paragraph"]
    drawing, photo = (200, 300, 800, 800), (200, 1100, 800, 1350)
    pics, fig_of, before = _picture_members(_page(ls, [drawing, photo], [drawing]), ls, kinds)
    assert [k for _, _, k in pics] == ["drawing", "photograph"]
    assert fig_of == {1: 0, 2: 0}  # the caption, outside the box, stays its own block
    assert before == {len(ls): [1]}  # the photograph is below every line: at the end of the page


def test_a_box_holding_running_text_is_not_taken_for_a_picture():
    prose = [L(100, 100 + 50 * k, 900, 140 + 50 * k, "سطر من النص الجاري في الفقرة الطويلة هنا") for k in range(4)]
    _, fig_of, _ = _picture_members(_page(prose, [(50, 50, 950, 400)]), prose, ["paragraph"] * 4)
    assert fig_of == {}


def test_a_picture_block_with_no_lines_has_a_page_and_a_box():
    b = Block([], "figure", picture=(3, (10, 20, 30, 40), "photograph"))
    assert b.page == 3 and b.box() == (10, 20, 30, 40) and b.text == ""


def test_an_equation_mostly_of_words_keeps_its_text_in_view():
    assert shows_words("خسارة الموازنة = عدد المحطات X وقت الدورة")
    assert not shows_words("ct = AT / OR ....(1)")
    assert not shows_words("N = Σ ti / Cd ...(2)")


def test_a_crop_reuses_the_page_image():
    from ocrl.doc.viewer import _crop

    h = _crop({"page": 5, "box": [1.0, 2.0, 3.0, 2.5], "kind": "formula"}, {"width": 8.0, "height": 10.0})
    assert 'class="crop pi5"' in h and "data:" not in h and "width:300px" in h
