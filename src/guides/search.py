from langchain_openai import OpenAIEmbeddings
from scipy.spatial.distance import cosine
import pandas as pd

df = pd.read_csv("echr_case_law_guides_with_openai_embeddings.csv")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

OPENAI_EMBEDDINGS = "openai_embeddings"
df[OPENAI_EMBEDDINGS] = df[OPENAI_EMBEDDINGS].apply(lambda x: eval(x))


def get_top_n_similarities(
    question: str, n: int = 5, desired_guide_ids: list[str] = []
):
    q_embedding = embeddings.embed_query(question)
    df_copy = df.copy()
    if desired_guide_ids:
        df_copy = df_copy[df_copy["guide_id"].isin(desired_guide_ids)]

    df_copy["similarity"] = df_copy[OPENAI_EMBEDDINGS].apply(
        lambda x: 1 - cosine(x, q_embedding)
    )

    return df_copy.nlargest(n, "similarity")
