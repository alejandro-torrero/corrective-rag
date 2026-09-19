from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocument, retrieval_grade
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str,Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("-- Check document relevance to question --")

    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        score: GradeDocument = retrieval_grade.invoke(
            {"question": question, "document": doc.page_content}
        )

        grade = score.binary_score

        if grade == "yes":
            print("Relevant document found")
            filtered_docs.append(doc)
        else:
            print("Irrelevant document found")
            web_search = True
            continue

        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }
