import pandas as pd
from langchain_openai import OpenAIEmbeddings

df = pd.read_csv("echr_case_law_guides.csv")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
paragraphs = df["paragraph"].tolist()
embeddings = embeddings.embed_documents(paragraphs)
df["openai_embeddings"] = embeddings
df.to_csv("echr_case_law_guides_with_openai_embeddings.csv", index=False)
