from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate

from guides.guides import (
    Guide,
    get_paragraphs,
    get_sentences,
    numbered_paragraphs_string,
)
from guides.search import get_top_n_similarities
from output_parsers import (
    extract_chosen_sentences,
    extract_chosen_paragraphs,
    extract_citations,
    extract_question,
    extract_question_simple,
)
from qg_prompts import (
    QG_SENTENCE_LEVEL_COT,
    QG_SINGLE_PARAGRAPH,
    QG_MULTIPLE_PARAGRAPHS,
)
from utils import numbered_string

llm = ChatOpenAI(model="gpt-3.5-turbo-16k")


def question_generation_sp(guide: Guide, paragraph: int):
    """
    Uses a single prompt to generate a question-answer pair with a single paragraphs.
    It defines a question that can be answered exactly with the provided paragraph.
    """
    answer = get_paragraphs(guide, [paragraph])
    prompt = PromptTemplate.from_template(QG_SINGLE_PARAGRAPH).format(paragraph=answer)
    response = llm.invoke(prompt)
    question = extract_question(response)
    return question, answer


def question_generation_mp(guide: Guide, paragraphs: list[int]):
    """
    Uses a single prompt to generate a question-answer pair with multiple paragraphs.
    It defines a question that can be answered exactly with the provided paragraphs.
    """
    paragraphs_str = numbered_paragraphs_string(guide, paragraphs)
    prompt = PromptTemplate.from_template(QG_MULTIPLE_PARAGRAPHS).format(
        paragraphs=paragraphs_str
    )
    response = llm.invoke(prompt)
    print("RESPONSE:", response.content)
    question = extract_question(response)
    chosen_paragraphs_indices = extract_chosen_paragraphs(response)
    print("CHOSEN INDICES:", chosen_paragraphs_indices)
    chosen_paragraphs_indices = [paragraphs[i - 1] for i in chosen_paragraphs_indices]
    print("CHOSEN PARAGRAPHS:", chosen_paragraphs_indices)
    answer = get_paragraphs(guide, chosen_paragraphs_indices)
    return question, answer, chosen_paragraphs_indices


def question_generation_cot(guide: Guide, paragraphs: list[int]):
    """
    Uses a single prompt to generate a question-answer pair with chain of thought.
    It defines a question that can be answered exactly with a subset of the provided sentences from the case law guide.
    It then provides the sentences it determines to be required to answer the question.
    """
    sentences = get_sentences(guide, paragraphs)
    prompt = PromptTemplate.from_template(QG_SENTENCE_LEVEL_COT).format(
        sentences=numbered_string(sentences)
    )
    response = llm.invoke(prompt)
    question = extract_question(response)
    chosen_sentences_indices = extract_chosen_sentences(response)
    chosen_sentences = [sentences[i - 1] for i in chosen_sentences_indices]
    answer = " ".join(chosen_sentences)
    return question, answer


def question_generation_with_search(
    guide: Guide, paragraphs: list[int], qg_prompt: str, answer_prompt: str
):
    """
    Uses two prompts to generate a question-answer pair with question to paragraph similarity search.
    The qg_prompt uses COT to generate a question based on the provided paragraphs.
    Using semantic search, we aim to include potentially relevant paragraphs for the answer.
    The answer_prompt generates an answer with citations based on the question and the paragraphs.
    We then use these citations to determine the final question-answer pair.
    """
    prompt = PromptTemplate.from_template(qg_prompt).format(
        paragraphs=get_paragraphs(guide, paragraphs)
    )
    response = llm.invoke(prompt)
    question = extract_question_simple(response)

    top_5 = get_top_n_similarities(question, 5, [guide.value])
    top_5_indices = set(top_5["paragraph_id"].tolist())
    top_5_and_paragraph_indices = list(set(paragraphs).union(top_5_indices))

    sentences = get_sentences(guide, top_5_and_paragraph_indices)
    prompt = PromptTemplate.from_template(answer_prompt).format(
        sentences=numbered_string(sentences), question=question
    )
    response = llm.invoke(prompt)

    cited_sentences_indices = extract_citations(
        response.content, set(range(1, len(sentences) + 1))
    )
    chosen_sentences = [sentences[i - 1] for i in cited_sentences_indices]
    answer = " ".join(chosen_sentences)
    return question, answer
