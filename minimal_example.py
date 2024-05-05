from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import pandas as pd

df = pd.read_csv("annotated_qa_pairs.csv")

## take all the pairs that have comprehensiveness score of 5 and preciseness score of 5
df = df[(df["comprehensiveness"] == 5) & (df["preciseness"] == 5)]

## now we only need question and answer columns
df = df[["question", "answer"]]

df = df.reset_index()

llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

result_df = pd.DataFrame(columns=["question", "answer", "generated_answer"])

PROMPT_TEMPLATE = """
Answer the following questions with citations from the HUDOC database.
Example citations: (Medvedyev and Others v. France [GC], 2010, § 76; Ladent v. Poland , 2008, § 45) or (Kudła v. Poland [GC], 2000, § 152)
Each sentence must be cited with the case name, year, and paragraph number!

Question: {question1}
Answer: {answer1}

Question: {question2}
Answer: {answer2}

Question: {question3}
"""

for i, row in df.iterrows():
    ## get the questions from the previous two rows
    question1 = df.loc[(i - 2) % len(df), "question"]
    answer1 = df.loc[(i - 2) % len(df), "answer"]
    question2 = df.loc[(i - 1) % len(df), "question"]
    answer2 = df.loc[(i - 1) % len(df), "answer"]
    question = row["question"]

    prompt = PROMPT_TEMPLATE.format(
        question1=question1,
        answer1=answer1,
        question2=question2,
        answer2=answer2,
        question3=question,
    )
    response = llm.invoke(prompt)
    response = response.content.removeprefix("Answer: ")

    print(response + "\n\n")

    result_df = result_df._append(
        {
            "question": question,
            "answer": row["answer"],
            "generated_answer": response,
        },
        ignore_index=True,
    )

    result_df.to_csv("generated_answers_example.csv", index=False)
