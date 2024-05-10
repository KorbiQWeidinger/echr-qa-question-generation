import unittest
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from citation_extraction.src.citation_extractor import (
    Citation,
    CitationExtractor,
    QAPair,
)

citation_extractor = CitationExtractor()


class TestCitationExtractor(unittest.TestCase):

    def test_example_1(self):
        qa_pair = QAPair(
            question="",
            answer="(McCann and Others v. the United Kingdom, 1995, § 202). Even though the Co",
            possible_citations={},
            guide_id="",
            paragraph_numbers=[],
        )
        expected = [
            Citation(
                case_name="McCann and Others v. the United Kingdom".lower(),
                best_match="McCann and Others v. the United Kingdom".lower(),
                case_id="",
                paragraph_numbers=[202],
            )
        ]
        result, errors = citation_extractor.extract_citations(qa_pair)
        self.assertEqual(result, expected)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
