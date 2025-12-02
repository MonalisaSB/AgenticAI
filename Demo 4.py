import streamlit as st
import autogen
import openai
from dotenv import load_dotenv
import os

client = openai.AzureOpenAI(
    api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",  
    api_version="2023-12-01-preview",
    azure_endpoint= "https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview" 
)

# Step 2: Define an IT Support chatbot with customizable behavior
class CustomBehaviorITSupportBot(autogen.AssistantAgent):
    def __init__(self, name, model="gpt-4o-mini", response_style="detailed", troubleshooting_priority="basic"):
        """
        Initializes the chatbot with:
        - A name for identification.
        - A selected GPT model (default: GPT-3.5 Turbo).
        - A response style that can be modified (e.g., 'detailed', 'concise', 'formal', 'casual').
        - A troubleshooting priority that determines whether to start with 'basic' or 'advanced' solutions.
        """
        super().__init__(name=name)
        self.model = model
        self.response_style = response_style  # Customize how the bot replies
        self.troubleshooting_priority = troubleshooting_priority  # Determines problem-solving approach

    def generate_reply(self, message):
        """
        Step 3: Handles user queries while considering customized behavior.
        - Adjusts response style (detailed, concise, formal, casual).
        - Prioritizes troubleshooting solutions based on user preferences.
        - Dynamically adapts responses based on issue complexity.
        """
        response = self._get_gpt_response(message)
        return response

    def _get_gpt_response(self, message):
        """
        Step 4: Uses OpenAI's GPT to generate a response based on:
        - Response style (how information is presented).
        - Troubleshooting priority (whether to start with basic or advanced solutions).
        - Adaptability to user feedback.
        """

        prompt = f"""
        The user reported an IT issue: "{message}". 
        You are an IT support assistant with the following behavior settings:
        - Response style: {self.response_style}
        - Troubleshooting priority: {self.troubleshooting_priority}

        Follow these guidelines:
        1. If troubleshooting priority is 'basic', suggest simple fixes first before advanced solutions.
        2. If response style is 'concise', keep responses under 3 sentences.
        3. If response style is 'formal', maintain professionalism in wording.
        4. If response style is 'casual', use a friendly and relaxed tone.
        5. Adapt troubleshooting steps dynamically based on user feedback.
        """

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT support assistant that adapts to user preferences and troubleshooting needs."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

# Step 5: Deploy using Streamlit
st.title("AI-Powered IT Support Chatbot")
st.write("Customizable AI assistant that adapts response style and troubleshooting approach.")

# User settings
response_style = st.selectbox("Select Response Style:", ["detailed", "concise", "formal", "casual"])
troubleshooting_priority = st.selectbox("Select Troubleshooting Priority:", ["basic", "advanced"])

# Initialize chatbot with selected settings
bot = CustomBehaviorITSupportBot(name="AdaptiveHelpBot", response_style=response_style, troubleshooting_priority=troubleshooting_priority)

# User input section
user_input = st.text_area("Enter your IT issue:")

if st.button("Get Help"):
    if user_input.strip():
        response = bot.generate_reply(user_input)
        st.subheader("AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a valid IT issue.")

