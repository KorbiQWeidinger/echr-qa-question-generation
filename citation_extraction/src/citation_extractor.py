import json
import re
import pandas as pd
import spacy
from pydantic import BaseModel

from citation_extraction.src.split_at_case_name import split_at_case_name
from citation_extraction.src.find_paragraphs import find_paragraphs
from citation_extraction.src.fuzzy_find import fuzzy_find_best, normalize_case_name

nlp = spacy.load("en_core_web_trf")

UNKNOWN_CITATION = "UNKNOWN_CITATION"


def get_sentences_spacy(text: str):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


class Paragraph(BaseModel):
    text: str
    paragraph_number: int
    possible_citations: dict[str, str]


class Citation(BaseModel):
    case_name: str
    best_match: str
    case_id: str
    paragraph_numbers: list[int]  # paragraph numbers the citation references
    snippet: str
    paragraphs: dict[int, str] | None = None


class Sentence(BaseModel):
    sentence: str
    citations: list[Citation]


class CitationNotIdentifiableException(Exception):
    def __init__(self, text: str):
        super().__init__(f"Could not identify citation {text}")


def get_snippet(text: str) -> str:
    i = text.find("v.")
    citation_area = text[max(0, i - 50) : min(i + 50, len(text) - 1)]
    return citation_area


def print_citation_area(text: str, guide_id: str) -> None:
    print(f"{guide_id} - Snippet with unidentifiable case name:", get_snippet(text))


def clean_text(text: str) -> str:
    pattern = r"Article \d+ § \d+"
    cleaned_text = re.sub(pattern, "", text)
    pattern = r"Rule (.*?) § \d+"
    cleaned_text = re.sub(pattern, "", cleaned_text)
    return cleaned_text


def extract_citations(
    text: str,
    possible_citations: dict[str, str],
    last_cited_case: str,
    last_cited_case_id: str,
    last_cited_case_snippet: str,
    manual_mappings: dict[str, str],
    guide_id: str,
):
    possible_citations.update(manual_mappings)
    sentences = get_sentences_spacy(text)

    possible_citations = {
        normalize_case_name(v): k for k, v in possible_citations.items()
    }

    sentences_with_citations: list[Sentence] = []

    for sentence in sentences:
        original_sentence = sentence
        sentence = clean_text(sentence)
        substrings_with_cases = split_at_case_name(sentence)

        if not substrings_with_cases:
            # special case: if there are paragraph numbers in the sentence but no case reference
            paragraph_numbers = find_paragraphs(sentence)
            if not paragraph_numbers:
                # sentence has no citation
                # ! We do not need this as we just use it without citation
                continue
            # then the previous citation is still valid
            sentences_with_citations.append(
                Sentence(
                    sentence=original_sentence,
                    citations=[
                        Citation(
                            case_name=last_cited_case,
                            best_match=last_cited_case,
                            case_id=last_cited_case_id,
                            paragraph_numbers=paragraph_numbers,
                            snippet=get_snippet(original_sentence),
                        )
                    ],
                )
            )
            continue

        citations = []

        # in the first substring with cases, if there are paragraphs before the first case reference, they are part of the last cited case
        paragraph_numbers = find_paragraphs(substrings_with_cases[0].split("v.")[0])
        if paragraph_numbers:
            citations.append(
                Citation(
                    case_name=last_cited_case,
                    best_match=last_cited_case,
                    case_id=last_cited_case_id,
                    paragraph_numbers=paragraph_numbers,
                    snippet=last_cited_case_snippet,
                )
            )

        for swc in substrings_with_cases:
            _, best_match, case_name = fuzzy_find_best(swc, possible_citations.keys())
            if case_name is None:
                print_citation_area(swc, guide_id)
                case_name = UNKNOWN_CITATION
                best_match = UNKNOWN_CITATION
                case_id = UNKNOWN_CITATION
            else:
                case_id = possible_citations[case_name]
            paragraph_numbers = find_paragraphs(swc.split("v.")[1])
            last_cited_case = case_name
            last_cited_case_id = case_id
            last_cited_case_snippet = get_snippet(swc)
            citations.append(
                Citation(
                    case_name=case_name,
                    best_match=best_match,
                    case_id=case_id,
                    paragraph_numbers=paragraph_numbers,
                    snippet=last_cited_case_snippet,
                )
            )
        sentences_with_citations.append(
            Sentence(
                sentence=original_sentence,
                citations=citations,
            )
        )

    return (
        sentences_with_citations,
        last_cited_case,
        last_cited_case_id,
        last_cited_case_snippet,
    )
