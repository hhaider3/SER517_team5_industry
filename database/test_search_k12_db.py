"""
test_search_k12_db.py
Unit tests for search_k12_db.py — tests pure logic functions without touching the real database.
"""

import unittest
from search_k12_db import clean_html, to_int, build_where, grade_penalty, combine_score, fmt_range


class TestCleanHtml(unittest.TestCase):
    def test_removes_html_tags(self):
        """HTML tags should be stripped from the string."""
        self.assertEqual(clean_html("<b>Hello</b>"), "Hello")

    def test_handles_none(self):
        """None input should return empty string."""
        self.assertEqual(clean_html(None), "")


class TestToInt(unittest.TestCase):
    def test_converts_valid_int(self):
        """Valid integer string should convert correctly."""
        self.assertEqual(to_int("5"), 5)

    def test_returns_default_on_invalid(self):
        """Invalid value should return the default."""
        self.assertIsNone(to_int("abc"))

    def test_returns_custom_default(self):
        """Invalid value should return custom default if provided."""
        self.assertEqual(to_int("abc", default=0), 0)


class TestBuildWhere(unittest.TestCase):
    def test_returns_none_with_no_filters(self):
        """No filters should return None."""
        self.assertIsNone(build_where(None, None, False))

    def test_subject_filter_only(self):
        """Subject filter should return a subject clause."""
        result = build_where("Math", None, False)
        self.assertEqual(result, {"subject": "Math"})

    def test_strict_grade_adds_clauses(self):
        """Strict grade should add grade_min and grade_max clauses."""
        result = build_where(None, 5, True)
        self.assertIn("$and", result)


class TestGradePenalty(unittest.TestCase):
    def test_no_penalty_when_grade_in_range(self):
        """No penalty when grade is within the image grade range."""
        self.assertEqual(grade_penalty(5, 4, 6), 0.0)

    def test_penalty_when_grade_outside_range(self):
        """Penalty should be applied when grade is outside range."""
        self.assertGreater(grade_penalty(10, 4, 6), 0.0)

    def test_no_penalty_when_grade_is_none(self):
        """No penalty when grade is not specified."""
        self.assertEqual(grade_penalty(None, 4, 6), 0.0)


class TestCombineScore(unittest.TestCase):
    def test_lower_distance_gives_higher_score(self):
        """Lower distance should produce a higher score."""
        score_low = combine_score(0.1, 0.0)
        score_high = combine_score(0.9, 0.0)
        self.assertGreater(score_low, score_high)

    def test_penalty_reduces_score(self):
        """Higher penalty should reduce the score."""
        score_no_penalty = combine_score(0.5, 0.0)
        score_with_penalty = combine_score(0.5, 0.5)
        self.assertGreater(score_no_penalty, score_with_penalty)


class TestFmtRange(unittest.TestCase):
    def test_formats_grade_range(self):
        """Should return formatted grade range string."""
        self.assertEqual(fmt_range(3, 5), "3 - 5")

    def test_returns_na_when_both_none(self):
        """Should return N/A when both values are None."""
        self.assertEqual(fmt_range(None, None), "N/A")

if __name__ == "__main__":
    unittest.main()