from ocrl.pipeline import EXTRA, MISSING, TOUCHING, check


def test_consistent_reading_has_no_findings():
    assert check("كتا", 1) == [] and check("كتا", 2) == []
    assert check("محمد", 0) == []


def test_dot_count():
    assert check("كتا", 0) == [MISSING]
    assert check("محمد", 1) == [EXTRA]
    assert check("شي", 6) == [EXTRA]


def test_touching_units_are_reported_separately_from_dot_count():
    assert check("كتاب", 2) == [TOUCHING]  # alef mid-body: two units, dots accounted for
    assert check("12", 0) == [TOUCHING]
