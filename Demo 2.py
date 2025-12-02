import autogen
import openai
import streamlit as st
from dotenv import load_dotenv
import os

# Step 1: Load environment variables (API keys)
client = openai.AzureOpenAI(
    api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",  
    api_version="2023-12-01-preview",
    azure_endpoint="https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview" 
)

# Step 2: Define an IT Support chatbot that remembers past issues
class ITSupportBot(autogen.AssistantAgent):
    def __init__(self, name, memory=None, model="gpt-4o-mini"):  
        super().__init__(name=name)
        self.memory = memory if memory is not None else {}  # Stores past user issues
        self.model = model  # Specifies GPT model

    def generate_reply(self, message, sender, **kwargs):
        """
        Step 3: Generates a response based on user input and past issues.
        - Retrieves past issues if available
        - Stores the latest issue in memory
        - Calls the GPT model to generate a response
        """
        context = self.memory.get(sender, "")  # Retrieves past issue if available
        self.memory[sender] = message  # Stores latest issue in memory
        
        response = self._get_gpt_response(message, context)  # Calls GPT for reply
        return response
    
    def _get_gpt_response(self, message, context):
        """
        Step 4: Calls OpenAI's GPT model to generate a response with past issue history.
        - Constructs a prompt including past conversation history
        - Sends the prompt to GPT-3.5 Turbo
        - Returns the generated response
        """
        prompt = f"User's previous issue: {context}\nNew issue: {message}\nIT Support Response:"
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful IT support assistant providing troubleshooting steps."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

# Step 5: Streamlit UI for real-time chatbot interaction
st.title(" IT Support Chatbot")
st.write("Ask me about your IT issues, and I'll provide troubleshooting steps!")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ITSupportBot(name="HelpDeskBot")

# Input field for user query
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = st.session_state.chatbot.generate_reply(user_input, "User1")
        st.write(f"**HelpDeskBot:** {response}")
