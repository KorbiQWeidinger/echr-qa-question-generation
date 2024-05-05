import re
from langchain_core.messages import AIMessage


def extract_question(message: AIMessage):
    match = re.search(r"Question: (.*)?", message.content)
    if match:
        return match.group(1)
    raise ValueError("No question found in the message", message.content)


def extract_question_simple(message: AIMessage):
    question = message.content.split("?")[0] + "?"
    question = question.replace("Question: ", "")
    return question


def extract_chosen_paragraphs(message: AIMessage):
    pattern = r"Paragraphs:.*?(\[.*\])"
    numbers = []
    matches = re.findall(pattern, message.content)
    for match in matches:
        numbers.extend(re.findall(r"\d+", match))
    chosen_paragraphs = [int(number) for number in numbers]
    if not chosen_paragraphs:
        raise ValueError("No chosen paragraphs found in the message", message.content)
    return chosen_paragraphs


def extract_chosen_sentences(message: AIMessage):
    pattern = r"Chosen Sentences:.*?(\[.*\])"
    numbers = []
    matches = re.findall(pattern, message.content)
    for match in matches:
        numbers.extend(re.findall(r"\d+", match))
    chosen_sentences = [int(number) for number in numbers]
    if not chosen_sentences:
        raise ValueError("No chosen sentences found in the message", message.content)
    return chosen_sentences


def extract_citations(text: str, allowed: set[int] = set()):
    pattern = r"\[\d+(?:-\d+)?(?:, \d+(?:-\d+)?)*\]"
    all_ints = set()
    matches = re.findall(pattern, text)
    for match in matches:
        items = re.findall(r"\d+(?:-\d+)?", match)
        for item in items:
            if "-" in item:
                start, end = map(int, item.split("-"))
                nums = range(start, end + 1)
            else:
                nums = [int(item)]
            for num in nums:
                if not allowed or num in allowed:
                    all_ints.add(num)
    return all_ints
