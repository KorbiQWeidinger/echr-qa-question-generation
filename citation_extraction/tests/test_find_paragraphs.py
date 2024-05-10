import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from citation_extraction.src.find_paragraphs import find_paragraphs


class TestFindParagraphs(unittest.TestCase):

    def test_single_paragraph(self):
        self.assertEqual(find_paragraphs("Refer to paragraph § 18"), [18])

    def test_range_paragraphs(self):
        self.assertEqual(find_paragraphs("See details in §§ 34-37"), [34, 35, 36, 37])

    def test_multiple_references(self):
        self.assertEqual(
            find_paragraphs("Details in §§ 87-89 and § 101"), [87, 88, 89, 101]
        )

    def test_multiple_ranges(self):
        self.assertEqual(
            find_paragraphs("Refer to §§ 1-3 and §§ 87-89"), [1, 2, 3, 87, 88, 89]
        )

    def test_no_paragraphs(self):
        self.assertEqual(find_paragraphs("There are no sections here"), [])

    def test_and_range(self):
        self.assertEqual(find_paragraphs("Refer to §§ 1-3 and 5"), [1, 2, 3, 5])

    def test_and_range_2(self):
        self.assertEqual(
            find_paragraphs("Refer to §§ 1-3 and 5, 6 foo 7"), [1, 2, 3, 5, 6]
        )


if __name__ == "__main__":
    unittest.main()
