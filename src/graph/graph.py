from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("Assess graded documents")

    if state["web_search"]:
        print("Decision: Not all documents are not relevant to question")
        return WEBSEARCH
    else:
        print("Decision: Generate")
        return GENERATE


def grade_generation_grounded_in_documents_and_questions(state: GraphState) -> str:
    print("Check hallucinations")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("Generation is grouneded in the documents")
        score = answer_grader.invoke({"question": question, "generation": generation})

        if answer_grade := score.binary_score:
            print("Valid answer")
            return "useful"
        else:
            print("Decision: answer is not useful")
            return "not useful"
    else:
        print("Decision: generation is not grounded")
        return "not supported"
    
def route_question(state: GraphState) -> str:
    print("Route question")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE



flow = StateGraph(GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)


flow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

flow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_questions,
    path_map={"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
)



flow.add_edge(WEBSEARCH, GENERATE)

flow.add_edge(GENERATE, END)


app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
