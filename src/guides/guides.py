import pandas as pd
from enum import Enum
import spacy

nlp = spacy.load("en_core_web_trf")


def get_sentences_spacy(text: str):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]


guides_df = pd.read_csv("echr_case_law_guides_with_openai_embeddings.csv")


class Guide(Enum):
    GUIDE_ART_1_ENG = "guide_art_1_eng"
    GUIDE_ART_2_ENG = "guide_art_2_eng"
    GUIDE_ART_3_ENG = "guide_art_3_eng"
    GUIDE_ART_4_ENG = "guide_art_4_eng"
    GUIDE_ART_5_ENG = "guide_art_5_eng"
    GUIDE_ART_6_CIVIL_ENG = "guide_art_6_civil_eng"
    GUIDE_ART_6_CRIMINAL_ENG = "guide_art_6_criminal_eng"
    GUIDE_ART_7_ENG = "guide_art_7_eng"
    GUIDE_ART_8_ENG = "guide_art_8_eng"
    GUIDE_ART_9_ENG = "guide_art_9_eng"
    GUIDE_ART_10_ENG = "guide_art_10_eng"
    GUIDE_ART_11_ENG = "guide_art_11_eng"
    GUIDE_ART_12_ENG = "guide_art_12_eng"
    GUIDE_ART_13_ENG = "guide_art_13_eng"
    GUIDE_ART_14_ART_1_PROTOCOL_12_ENG = "guide_art_14_art_1_protocol_12_eng"
    GUIDE_ART_15_ENG = "guide_art_15_eng"
    GUIDE_ART_17_ENG = "guide_art_17_eng"
    GUIDE_ART_18_ENG = "guide_art_18_eng"
    ADMISSIBILITY_GUIDE_ENG = "Admissibility_guide_ENG"
    GUIDE_ART_46_ENG = "guide_art_46_eng"
    GUIDE_ART_1_PROTOCOL_1_ENG = "guide_art_1_protocol_1_eng"
    GUIDE_ART_2_PROTOCOL_1_ENG = "guide_art_2_protocol_1_eng"
    GUIDE_ART_3_PROTOCOL_1_ENG = "guide_art_3_protocol_1_eng"
    GUIDE_ART_2_PROTOCOL_4_ENG = "guide_art_2_protocol_4_eng"
    GUIDE_ART_3_PROTOCOL_4_ENG = "guide_art_3_protocol_4_eng"
    GUIDE_ART_4_PROTOCOL_4_ENG = "guide_art_4_protocol_4_eng"
    GUIDE_ART_1_PROTOCOL_7_ENG = "guide_art_1_protocol_7_eng"
    GUIDE_ART_2_PROTOCOL_7_ENG = "guide_art_2_protocol_7_eng"
    GUIDE_ART_4_PROTOCOL_7_ENG = "guide_art_4_protocol_7_eng"
    GUIDE_DATA_PROTECTION_ENG = "guide_data_protection_eng"
    GUIDE_ENVIRONMENT_ENG = "guide_environment_eng"
    GUIDE_IMMIGRATION_ENG = "guide_immigration_eng"
    GUIDE_MASS_PROTESTS_ENG = "guide_mass_protests_eng"
    GUIDE_PRISONERS_RIGHTS_ENG = "guide_prisoners_rights_eng"
    GUIDE_LGBTI_RIGHTS_ENG = "guide_lgbti_rights_eng"
    GUIDE_SOCIAL_RIGHTS_ENG = "guide_social_rights_eng"
    GUIDE_TERRORISM_ENG = "guide_terrorism_eng"


def get_guide(guide: Guide):
    df_copy = guides_df.copy()
    df_copy = df_copy[df_copy["guide_id"].isin([guide.value])]
    df_copy = df_copy.reset_index(drop=True)
    return df_copy


def get_paragraphs(guide: Guide, paragraphs: list[int]):
    df = get_guide(guide)
    paragraphs_df = df.loc[paragraphs]
    paragraphs_str = " ".join(paragraphs_df["paragraph"])
    return paragraphs_str


def numbered_paragraphs_string(guide: Guide, paragraphs: list[int]):
    df = get_guide(guide)
    paragraphs_df = df.loc[paragraphs]
    paragraphs_list = paragraphs_df["paragraph"].tolist()
    paragraphs_numbered_str = "\n".join(
        f"{i+1}. {paragraph}" for i, paragraph in enumerate(paragraphs_list)
    )
    return paragraphs_numbered_str


def get_sentences(guide: Guide, paragraphs: list[int]):
    paragraphs_str = get_paragraphs(guide, paragraphs)
    sentences = get_sentences_spacy(paragraphs_str)
    return sentences


def numbered_sentence_string(guide: Guide, paragraphs: list[int]):
    sentences = get_sentences(guide, paragraphs)
    sentences_with_numbers = "\n".join(f"[{i+1}]: {s}" for i, s in enumerate(sentences))
    return sentences_with_numbers
