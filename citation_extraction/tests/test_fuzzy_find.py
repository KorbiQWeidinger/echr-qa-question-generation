import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import unittest
from citation_extraction.src.fuzzy_find import (
    normalize_case_name,
    fuzzy_find_case_name,
    fuzzy_find_best,
)


class TestFuzzyFindMethods(unittest.TestCase):

    def test_normalize_case_name(self):
        case_name = "Case of Apple Against Banana"
        result = normalize_case_name(case_name)
        self.assertEqual(result, "apple v. banana")

    def test_fuzzy_find_case_name_exact_match(self):
        text = "apple v. banana cherry v. date"
        case_name = "apple v. banana"
        result = fuzzy_find_case_name(text, case_name)
        self.assertEqual(result, (0, "apple v. banana", 0))

    def test_fuzzy_find_case_name_partial_match(self):
        text = "apple v. banana cherry v. date"
        case_name = "abple v. banana"
        result = fuzzy_find_case_name(text, case_name)
        self.assertEqual(result[0], 0)

    def test_fuzzy_find_case_name_no_match(self):
        text = "apple v. banana cherry v. date"
        case_name = "orange v. pear"
        result = fuzzy_find_case_name(text, case_name)
        self.assertEqual(result, (-1, None, 0))

    def test_fuzzy_find_best(self):
        text = "appae v. banana foo"
        case_names = ["apple v. banana"]
        result = fuzzy_find_best(text, case_names)
        self.assertEqual(result, (0, "appae v. banana", "apple v. banana"))

    def test_fuzzy_find_best_2(self):
        text = "in the Rantsev v. Chypre and Russia judgment"
        case_names = ["Rantsev v. Chypre and Russia"]
        _, best_match, _ = fuzzy_find_best(text, case_names)
        self.assertEqual(best_match, "Rantsev v. Chypre and Russia".lower())

    def test_fuzzy_find_best_3(self):
        text = "(McCann and Others v. the United Kingdom, 1995, § 202)"
        case_names = ["McCann and Others v. the United Kingdom"]
        _, best_match, _ = fuzzy_find_best(text, case_names)
        self.assertEqual(best_match, case_names[0].lower())


if __name__ == "__main__":
    unittest.main()
