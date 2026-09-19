from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocument, retrieval_grade
from ingestion.ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent_memory"
    docs = retriever.invoke(question)

    doc_txt = docs[1].page_content

    res: GradeDocument = retrieval_grade.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retriebal_grader_answer_no() -> None:
    question = "agent_memory"
    docs = retriever.invoke(question)

    doc_txt = docs[1].page_content

    res: GradeDocument = retrieval_grade.invoke(
        {"question": "How to make pizza?", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})

    pprint(generation)
