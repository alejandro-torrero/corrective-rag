from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
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

    doc_txt = docs[1].page_content

    generation = generation_chain.invoke({"context": doc_txt, "question": question})

    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    joined_docs = "\n".join([d.page_content for d in docs])

    generation = generation_chain.invoke({"context": joined_docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"

    docs = retriever.invoke(question)

    joined_docs = "\n".join([d.page_content for d in docs])

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": joined_docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )

    assert not res.binary_score
