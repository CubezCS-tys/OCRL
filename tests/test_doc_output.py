from ocrl.doc.tables import innermost
from ocrl.doc.viewer import _dir


def test_a_frame_round_another_frame_is_not_a_table():
    page_border, table, other = (50, 40, 2400, 3400), (200, 900, 2200, 1800), (200, 2000, 2200, 2600)
    assert innermost([page_border, table, other]) == [table, other]
    assert innermost([table, other]) == [table, other]  # side by side: both stay


def test_block_direction_follows_its_script():
    assert _dir("الفصل الأول من الكتاب") == "rtl"
    assert _dir("The first chapter (1990) of the book") == "ltr"
    assert _dir("Les résultats de l'étude") == "ltr"
    assert _dir("COVID-19 وأثره على التعليم في المدارس") == "rtl"
    assert _dir("") == "rtl"


def test_a_table_part_over_the_page_reads_in_the_first_part_s_direction():
    from types import SimpleNamespace as L
    from ocrl.doc.tables import Cell, Table, _script_rtl, same_direction

    first = Table(11, (0, 0, 10, 10), [], 1, 3, rtl=False)
    cells = [Cell([L(text="٤٥")], 0, 0), Cell([L(text="NO")], 0, 1), Cell([L(text="Task 44")], 0, 2)]
    nxt = Table(12, (0, 0, 10, 10), cells, 1, 3, rtl=True)
    same_direction(first, nxt)
    assert not nxt.rtl and [c.text for c in nxt.cells] == ["Task 44", "NO", "٤٥"]
    # Arabic-Indic digits are not Arabic letters: a column of them beside English words is an English table
    assert not _script_rtl([L(text="Task 44"), L(text="١٫٦٩"), L(text="٤٥")], [])
    assert _script_rtl([L(text="المجموع"), L(text="٤٥")], [])


def test_viewer_links_each_line_to_its_place_and_lists_every_type(tmp_path):
    import numpy as np
    from ocrl.doc.viewer import write_viewer

    part = lambda texts: {"page": 1, "regions": [[1, 1, 7, 3]], "line_boxes": [[1, 1 + i, 7, 1.5 + i] for i in range(len(texts))], "line_texts": texts}
    master = {"id": "t", "made_by": {"text": "r", "structure": "s"}, "pages": [{"number": 1, "width": 8.0, "height": 11.0, "unit": "inch"}],
              "blocks": [{"id": "b0", "type": "verse", "text": "سطر أول سطر ثان", "parts": [part(["سطر أول", "سطر ثان"])], "order": 1},
                         {"id": "b1", "type": "paragraph", "text": "An English line", "parts": [part(["An English line"])], "order": 2}]}
    write_viewer(master, [np.full((110, 80), 255, np.uint8)], tmp_path / "v.html")
    page = (tmp_path / "v.html").read_text()
    assert page.count('class="lb"') == 3 and page.count('class="ln"') == 3
    assert "Verse · 1" in page  # a type with no colour of its own still has its place in the legend
    assert 'dir="ltr" lang="en"' in page and 'dir="rtl" lang="ar"' in page


def test_a_picture_read_as_many_figure_blocks_is_folded():
    from ocrl.doc.viewer import _fold_figures

    fig = lambda o: ({"order": o, "parts": [{"page": 1}]}, 1)
    items = [None, "<p>page 1</p>"] + sum(([fig(o), f'<div class="blk figure">{o}</div>'] for o in range(1, 5)), []) + [None, '<div class="blk paragraph">p</div>']
    out = _fold_figures(items)
    assert len(out) == 3 and out[1].startswith("<details") and "4 pieces" in out[1]
    assert _fold_figures(items[:6]) == ["<p>page 1</p>", '<div class="blk figure">1</div>', '<div class="blk figure">2</div>']
