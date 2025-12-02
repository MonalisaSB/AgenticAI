# Step 0: Import required libraries

from dotenv import load_dotenv
import os
import streamlit as st #ui
from pydantic import BaseModel
import openai

# Step 1: Load environment variables
# Load the OpenAI API key from a .env file to authenticate with the OpenAI API.

client = openai.AzureOpenAI(
    api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",  
    api_version="2025-01-01-preview",
#    azure_endpoint = "https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview"
    azure_endpoint = "https://openai-api-management-gw.azure-api.net/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
)

# Step 2: Define a structured schema using Pydantic
# This schema will ensure that the LLM output is structured and validated.
class WebSearchPrompt(BaseModel):
    search_query: str
    justification: str


class SearchSummary(BaseModel):
    search_query: str
    justification: str
    top_links:list
    summarized_answer: str

import requests
import re
def web_search_duckducgo(justification, max_results=5):
    cleaning = " ".join(re.findall(r"\b\w+\b", justification))
    q1 = cleaning.strip()[:100]
    url = "https://api.duckduckgo.com/"
    params = {"q":q1, "format":"json", "no_html":1}
    response = requests.get(url, params=params).json()

    results = []
    if "RelatedTopics" in response:
        for i in response["RelatedTopics"][:max_results]:
            if "Text" in i and "FirstURL" in i:
                results.append(f"{i["Text"]}-{i["FirstURL"]}")
    return results if results else ["No results found"]



# Step 3: Build Streamlit UI
# Use Streamlit to create a simple UI where users can input their question and get a structured response.
st.title("Web Search Optimization with LLM")
st.write("Enter a question to receive an optimized web search query and reasoning.")

#initialization memory
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Step 4: Create input field for the user's question
user_query = st.text_input("Enter your question:") # ask question



# Step 5: Process the input query and display the result
if user_query:
    st.info("Agent is thinking")
    plan_prompt  = f"""
    you are a intelligent web search agent for the given question {user_query}
    
    explain briefly"""
    # Invoke the LLM with the user query
    # Extract relevant parts of the response
    response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[{"role": "system", "content": "you are a smart planning agent"},
                                          {"role": "user", "content": plan_prompt}],
                                          temperature=0.3)
    response_content = response.choices[0].message.content

    # Structure the output using the pydantic model
    search_plan = WebSearchPrompt(search_query=user_query, justification=response_content)
    st.info("Searching the web")
    search_results = web_search_duckducgo(search_plan.justification)
    summary_prompt = f"""based the search 
    query: {search_plan.search_query}
    results: {search_results}
    please summarize the results"""
    summary_response = client.chat.completions.create(model="gpt-4o-mini",
                                          messages=[{"role": "system", "content": "you are a smart summarizing agent"},
                                          {"role": "user", "content": summary_prompt}],
                                          temperature=0.3)
    response_summ= summary_response.choices[0].message.content


    agent_output = SearchSummary(search_query=search_plan.search_query,
                                justification=search_plan.justification,
                                top_links=search_results,
                                summarized_answer=response_summ)

    # Display the structured response to the user
    st.subheader("Optimized User Query:")
    st.write(agent_output.search_query)  # Display the optimized search query
    st.subheader("Reasoning: (Model output)")
    st.write(agent_output.justification) 

    st.subheader("Top link")
    st.write(agent_output.top_links)   # Display the reasoning behind the query
    st.subheader("Summary")
    st.write(agent_output.summarized_answer) 