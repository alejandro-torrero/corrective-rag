from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODEL = "qwen3:8b"

llm = ChatOllama(model=MODEL, temperature=0)


class GradeDocument(BaseModel):
    """Binary Score for relevance check on retrieved documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocument)


system_prompt = """
You are a grading assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant
Give a binary score: yes or no; To indicate whether the doucment is relevant to the question
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Retrieved Document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grade = grade_prompt | structured_llm_grader
