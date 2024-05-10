def split_at_case_name(sentence: str) -> list[str]:
    """
    Very simple method to find areas with case names in a text.
    Splits the text at v. and joins the two neighboring sentences.
    """
    if " v. " not in sentence:
        return []
    if sentence.count(" v. ") == 1:
        return [sentence]
    strings_with_one_case = []
    splits = sentence.split(" v. ")
    for i in range(len(splits) - 1):
        strings_with_one_case.append(splits[i] + " v. " + splits[i + 1])

    return strings_with_one_case
