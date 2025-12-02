# Step 1: Set Up API Key
# Ensure you have set up your OpenAI API key before running this script.
# Recommended approach: Set the key as an environment variable.

import os
import autogen
from dotenv import load_dotenv


# Step 1: Define Customer and Support Agents
# Creating a customer agent that represents a user seeking support.

customer_agent = autogen.UserProxyAgent(
    name="customer",
    human_input_mode="ALWAYS",  # Allows manual input for demonstration purposes.
    code_execution_config = {"use_docker": False},
    max_consecutive_auto_reply= 5
)

# Creating a support agent that will respond to customer queries.
support_agent = autogen.AssistantAgent(
    name="support_agent",
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
    code_execution_config = {"use_docker": False},
    max_consecutive_auto_reply= 5

)

# Step 2: Configuring the Support Agent
# Modifying the support agent to follow a structured approach with clear and professional responses.
support_agent = autogen.AssistantAgent(
    name="support_agent",
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
    system_message="You are a helpful AI support agent. Answer customer queries clearly and professionally.",
    code_execution_config = {'use_docker': False},
    max_consecutive_auto_reply= 5
)

# Step 3: Running a Simulated Customer Interaction
# The customer initiates a conversation with the support agent.

customer_agent.initiate_chat(support_agent, message="I need help tracking my order.")

