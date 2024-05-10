import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from citation_extraction.src.split_at_case_name import split_at_case_name


class TestSplitAtCaseName(unittest.TestCase):

    def test_split_at_case_name_single_case(self):
        text = "Case of Apple v. Banana: Apple won."
        expected = [text]
        result = split_at_case_name(text)
        self.assertEqual(result, expected)

    def test_split_at_case_name_multiple_cases(self):
        text = "Case of Apple v. Banana: Apple won. Case of Cherry v. Date: Cherry won."
        result = split_at_case_name(text)
        self.assertEqual(
            result,
            [
                "Case of Apple v. Banana: Apple won. Case of Cherry",
                "Banana: Apple won. Case of Cherry v. Date: Cherry won.",
            ],
        )

    def test_split_at_case_name_no_match(self):
        text = "Case of Apple Against Banana: Apple won."
        result = split_at_case_name(text)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
