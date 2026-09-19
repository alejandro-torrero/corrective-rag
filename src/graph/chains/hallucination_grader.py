from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

MODEL = "qwen3:8b"

llm = ChatOllama(model=MODEL, temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    reason: str = Field(
        description="Describe the reason of your decision about the binary_score"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system_prompt = """
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.

If the generation has nothing to do with the set of facts, then it is an hallucination and the binary_score has to be False

If they are related, then binary_score must be True

For example, if the generation is about pizza and the facts are about LLMs, then its is an halluciantion, therefore the binary_score must be False

"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
