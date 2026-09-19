from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

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


flow = StateGraph(GraphState)

flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)

flow.set_entry_point(RETRIEVE)
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

flow.add_edge(WEBSEARCH, GENERATE)

flow.add_edge(GENERATE, END)


app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")