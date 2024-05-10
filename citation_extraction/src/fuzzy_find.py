import regex

MAX_ERRORS = 3


def normalize_case_name(case_name: str) -> str:
    return case_name.lower().replace("case of ", "").replace("against", "v.")


def fuzzy_find_case_name(text: str, case_name: str):
    text = text.lower()
    case_name = normalize_case_name(case_name)

    for errors_allowed in range(MAX_ERRORS + 1):
        pattern = f"({regex.escape(case_name)}){{e<={errors_allowed},s<=2}}"
        matches = regex.findall(pattern, text)
        if matches:
            case_name = matches[0]
            index = text.find(case_name)
            return index, case_name, errors_allowed
    return -1, None, 0


def fuzzy_find_best(text: str, case_names: list[str]):
    best_match = None
    best_match_index = -1
    best_match_errors = 0
    best_cn = None
    for cn in case_names:
        index, match, errors = fuzzy_find_case_name(text, cn)
        if index != -1 and (best_match is None or errors < best_match_errors):
            best_match = match
            best_match_index = index
            best_match_errors = errors
            best_cn = cn
    return best_match_index, best_match, best_cn
