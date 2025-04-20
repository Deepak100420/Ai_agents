from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gorq_api_key = os.getenv("GORQ_API_KEY")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=gorq_api_key
)

# Define State
class State(BaseModel): 
    input: str = ""
    post: str = ""
    critique: str = ""
    iteration: int = 0

# Initialize graph
graph = StateGraph(State)

# Generation Node
def generation_node(state: State) -> State: 
    template = [
        ("system", """You are a LinkedIn techie influencer assistant who writes posts.

        Always respond **only** with the final post content—no explanations, steps, or extra comments. If the user has given critique, revise the previous version accordingly."""),
        MessagesPlaceholder(variable_name="messages")
    ]

    prompt = ChatPromptTemplate.from_messages(template)
    response = prompt | llm

    messages = [HumanMessage(content=state.input)]
    if state.iteration > 0:
        messages.append(AIMessage(content=state.post))
        messages.append(HumanMessage(content=state.critique))

    result = response.invoke({"messages": messages}).content

    state.post = result
    state.iteration += 1
    return state

# Critique Node
def critique_node(state: State) -> State: 
    template = [
        ("system", """You are a social media content expert who evaluates LinkedIn posts.

Your job is to analyze the quality of any LinkedIn post based on these criteria:

1.Clarity of the message

2.Value delivered to the reader (insights, tips, or thought leadership)

3.Engagement potential (hooks, emotional appeal, or call-to-action)

4.Structure and formatting (line breaks, readability, flow)

5.Professional tone and relevance to the LinkedIn audience

If the post satisfies all criteria, respond with:
"Perfect"

If it does not, provide clear and constructive feedback on which areas need improvement."""),
        MessagesPlaceholder(variable_name="messages")
    ]

    prompt = ChatPromptTemplate.from_messages(template)
    response = prompt | llm

    result = response.invoke({"messages": [HumanMessage(content=state.post)]}).content
    state.critique = result
    return state

# Decision Node
def decision_node(state: State):
    if "Perfect" in state.critique:
        return END
    elif state.iteration > 3:
        return END
    else:
        return "generation"

# Add nodes and edges
graph.add_node("generation", generation_node)
graph.add_node("review", critique_node)

graph.set_entry_point("generation")
graph.add_edge("generation", "review")
graph.add_conditional_edges("review", decision_node)

# Compile the graph
app = graph.compile()

print("\n✨ Starting the generation and review process:\n")

final_state = None  # To capture the final state

user_input = "Write a LinkedIn post about the latest advancements in generative AI for e-commerce."

# Stream through the process
for output in app.stream({"input": user_input}):
    for key, value in output.items():
        if key == "generation":
            print(f"\n🧠 Generation Agent  (Generated Post - Iteration {value['iteration']}):\n{value['post']}\n{'-'*50}")
        elif key == "review":
            print(f"\n🧐 Critique Agent (Review - Iteration {value['iteration']}):\n{value['critique']}\n{'='*50}")
    final_state = value  # Update the final state

# Final output
if final_state:
    print("\n🎉 FINAL POST (After Iterations):")
    print(final_state['post'])
    print("--------------------")
    print("\nCritique:", final_state.get('critique', 'No critique'))
    print("--------------------")
    print(f"Total iterations: {final_state['iteration']}")
