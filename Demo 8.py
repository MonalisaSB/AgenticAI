import streamlit as st
from typing import TypedDict
from autogen import AssistantAgent, UserProxyAgent
from langgraph.graph import StateGraph, START

# Define the state for LangGraph
class QueryState(TypedDict):
    user_input: str
    response: str

# Define AutoGen Assistant Agent
define_assistant = AssistantAgent(
    name="FAQ_Bot",
    llm_config={
        "config_list": [
            {
                "api_type": "azure",
                "azure_endpoint": "https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview" ,
                "api_version": "2023-12-01-preview",
                "api_key": "2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",
                # "deployment_name": "gpt-4o-mini",  # corrected key name
                "model": "gpt-4o-mini",  # optional, but good to keep
            }
        ],
        "temperature": 0.7,
    },
    system_message="I can answer FAQs and check order statuses.",
    code_execution_config={'use_docker': False}
)

user_agent = UserProxyAgent(name="User", code_execution_config={'use_docker': False})

# Function to handle FAQs
def faq_handler(state: QueryState) -> QueryState:
    response = define_assistant.run(state["user_input"])  # Generate response
    return {"user_input": state["user_input"], "response": response}

# Function to escalate complex queries
def escalate_to_human(state: QueryState) -> QueryState:
    return {"user_input": state["user_input"], "response": "Escalating to human support. Please wait."}

# Create LangGraph Workflow
graph = StateGraph(QueryState)
graph.add_node("FAQ_Handler", faq_handler)
graph.add_node("Escalation", escalate_to_human)

graph.add_edge(START, "FAQ_Handler")    
graph.add_edge("FAQ_Handler", "Escalation")

workflow = graph.compile()


# Streamlit UI
st.title("Customer Support AI Chatbot")
st.write("Ask a question and let the AI assist you.")

user_input = st.text_input("Enter your query:")
if st.button("Submit"):
    query = QueryState(user_input=user_input)
    result = workflow.invoke(query)
    st.write("Response:", result.response)