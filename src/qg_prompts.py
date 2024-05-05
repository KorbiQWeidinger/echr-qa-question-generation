""" PROMPTS FOR QUESTION GENERATION """

QG_SINGLE_PARAGRAPH = """
Task: 
Define a single question based on the provided paragraph from the ECHR case law guides.
The paragraph must answer the question precisely and completely.
Think about the context of the paragraph and how it can be used to create a challenging legal question.
Use the language from the paragraph in the question.

Paragraph:  
{paragraph}

Answer Template:
Thoughts: {{thoughts on the paragraph and what question could be asked}}
Question: {{the question that can be answered exactly by the provided paragraph}}
"""

QG_MULTIPLE_PARAGRAPHS = """
Task: 
Define a single question based on the provided paragraphs from the ECHR case law guides.
The paragraphs must answer the question precisely and completely.
Think about the context of each paragraph and what challenging legal question it can answer.
Use the language from the paragraphs in the question.

Paragraphs:  
{paragraphs}

Answer Template:
Thoughts: {{thoughts on the paragraphs and what question could be asked}}
Question: {{the question that can be answered exactly by the provided paragraphs}}
Paragraphs: {{Array of paragraph numbers needed to answer the question. Example: [1,2,6,7]}}
"""

QG_SENTENCE_LEVEL_COT = """
Scenario: 
Assume the role of an experienced lawyer specializing in ECHR case law.
Your objective is to develop educational and challenging questions for trainee lawyers. 
Each question should be based on the provided paragraphs from the ECHR case law guides. 
When formulating a question avoid over-contextualizing (referring to specific countries or asking about specific cases) and avoid asking questions about broad concepts for which the provided sentences only give partial answers.

Sentences:  
{sentences}

Steps:
1. Come up with a question that can be answered exactly with a subset of the provided sentences from the case law guide.
2. For each sentence reason about if it is required to answer the question.
3. Formulate an exact answer by combining the provided sentences. 

Answer Template: 
Question: {{Question that can be answered exactly with a subset of the provided sentences}}
Sentence [1]: {{Reason about why this sentence is (not) required for the answer}}
Sentence [2]: {{Reason about why this sentence is (not) required for the answer}}
... (reason about all sentences 1 to k)
Sentence [k]: {{Reason about why this sentence is (not) required for the answer}}
Chosen Sentences: {{Array of chosen sentences. Example: [1,2,4,5,6,7]}}
"""

QG_WITH_SEARCH_1 = """
Case law paragraphs: 
{paragraphs}

Task:
Define a single challenging legal question that can be answered with the given case law paragraphs.
Reuse the language from the case law in the question.

Question: 
"""

GQ_WITH_SEARCH_2 = """
Case law sentences: 
{sentences}

Task: Answer the following question with the provided sentences.
At the end of each sentence in your answer cite the used sentences in square brackets.

Question: 
{question}
Answer: 
"""

QG_LEGAL_1_1 = """
Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts. 
Emphasize the patterns that link facts to specific legal doctrines.

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify how the margin of appreciation and positive obligations apply in relation to the State's discretion
2. Identify the reasons that justify necessity and pressing social needs
3. Identify the reasons that command that rights be effective in their application
4. Identify how reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals
5. Identify the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities
6. Define a question that can be answered exactly by the given legal doctrines and applicable facts to those doctrines

Answer Template:
Margin of appreciation: {{how do the margin of appreciation and positive obligations apply in relation to the State's discretion}}
Necessities: {{reasons that justify necessity and pressing social needs}}
Effectivity: {{reasons that command that rights be effective in their application}}
Reasonable Measures: {{how do reasonable measures apply in relation to the State's discretion and to restrictions imposed by States or private individuals?}}
Domestic Reasons: {{the reasons set forth by the Court to defer to domestic reasons provided by domestic authorities}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts reusing the language from the ECHR case law}}
"""

QG_LEGAL_1_2 = """
Your objective is to develop educational and challenging question-answer pairs for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.

Doctrines and facts: 
{sentences}

Task: Answer the following question based on the provided doctrines and facts from the ECHR case law guides.

Use the provided doctrines and facts to answer the question.
Use citations! At the end of each sentence in your answer add all the numbers of the used facts and doctrines in square brackets.

Question: 
{question}
Answer: 
"""

QG_LEGAL_2_1 = """
Your objective is to develop educational and challenging questions for lawyers working with ECHR case law and for judges who want to draft judgments based on ECHR case law.
Each question should be based on the provided paragraphs from the ECHR case law guides.
When formulating a question reuse the language from the ECHR case law and match legal doctrines to specific facts.
Emphasize the patterns that link facts to specific legal doctrines.

Doctrines and facts: 
{paragraphs}

Steps:
1. Identify what are the criteria under the Convention for applying the rights enshrined therein.
2. Identify the conditions that the Court sets forth with view to analyse the legality of domestic measures and restrictions
3. Identify the reasons provided by the Court to protect applicants and victims and to differentiate between them.
4. Identify the reasons set forth by the Court to distinguish between legal doctrines and contextual application of those doctrines.
5. Assign a set of facts to its corresponding Article and identify a sequence of reasons that justify the application of the Article to those facts.
6. Explain why analogies and comparisons between Article-fact pair are pertinent
7. Identify separately reasons that are linked to margin of appreciation of the State from those linked to the Court's appreciation of facts
8. Define a question that can be answered exactly by the given Article-facts correspondence

Answer Template:
Criteria for rights: {{how does the Court define the criteria for applying rights}}
Legality of domestic measures and restrictions: {{conditions that determine that domestic measures are compliant with the Convention}}
Protection and differentiation of applicants and victims: {{circumstances and conditions that limit or allow applicants and victims to present their case}}
Distinction between legal doctrines and contextual application: {{how and why and in what circumstances legal doctrines apply to specific facts}}
Article-facts correspondence: {{the reasons set forth by the Court to justify the application of the Article/Articles to those facts}}
Analogies and comparisons between Article-fact pair {{the reasons set forth by the Court to justify why articles and facts differ from one another}}
margin of appreciation of States and Court's appreciation {{the reasons, circumstances and conditions set forth by the Court to explain margin of appreciation of States and its own appreciation}}
Question: {{define a single question that can be answered exactly by the given legal doctrines and applicable facts and by the Article-fact pair, reusing the language from the ECHR case law and adjusting the question and the answer depending on how fact-Article correspondence is better addressed with What? How? Why?}}
"""

QG_LEGAL_2_2 = """
Your objective is to develop question-answer pairs that could help lawyers working with ECHR case law 
and judges who want to draft judgments based on ECHR find the best arguments to justify violations or non-violations of the ECHR.

Doctrines, articles, and facts:
{paragraphs}

Answer the following question following the sequence: characterization of facts according to ECHR doctrines and justification of those facts in relation to specific articles of the ECHR.
Use the provided doctrines and facts to answer the question.
Use citations! At the end of each sentence in your answer add the numbers of the used facts and doctrines in square brackets.

Question: 
{question}
Answer: 
"""
