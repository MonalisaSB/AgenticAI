import streamlit as st
import autogen
import openai
from dotenv import load_dotenv
import os

client = openai.AzureOpenAI(
    api_key="2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC",  
    api_version="2023-12-01-preview",
    azure_endpoint="https://openai-api-management-gw.azure-api.net/openaiprodtest/deployments/gpt-4o-mini/chat/completions?api-version=2023-12-01-preview" 
)

# Define the Diagnostic Agent
class DiagnosticAgent(autogen.AssistantAgent):
    def __init__(self, name="DiagnosticAgent"):
        super().__init__(name=name)

    def analyze_issue(self, issue):
        """Analyzes the issue and decides whether to resolve or escalate."""
        if any(keyword in issue.lower() for keyword in ["password", "slow internet", "software update"]):
            return "simple"
        elif any(keyword in issue.lower() for keyword in ["network failure", "hardware crash", "security breach"]):
            return "critical"
        else:
            return "complex"

# Define the Resolution Agent
class ResolutionAgent(autogen.AssistantAgent):
    def __init__(self, name="ResolutionAgent", model="gpt-4o-mini"):
        super().__init__(name=name)
        self.model = model

    def resolve_issue(self, issue):
        """Uses GPT to troubleshoot complex problems."""
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an IT support specialist providing solutions to technical problems."},
                {"role": "user", "content": f"Troubleshoot the following IT issue: {issue}"}
            ]
        )
        return response.choices[0].message.content.strip()

# Define the Escalation Agent
class EscalationAgent(autogen.AssistantAgent):
    def __init__(self, name="EscalationAgent"):
        super().__init__(name=name)

    def escalate_issue(self, issue):
        """Escalates the issue to human IT support."""
        return f"The issue '{issue}' requires human intervention. Escalating to IT support."

# Streamlit UI
st.title("Multi-Agent IT Support System")
st.write("AI-powered IT support with multi-agent interaction.")

# User Input
user_issue = st.text_area("Describe your IT issue:")

if st.button("Get Support"):
    if user_issue.strip():
        diagnostic_agent = DiagnosticAgent()
        resolution_agent = ResolutionAgent()
        escalation_agent = EscalationAgent()

        issue_type = diagnostic_agent.analyze_issue(user_issue)

        if issue_type == "simple":
            st.success("The issue has been resolved with basic troubleshooting.")
        elif issue_type == "complex":
            response = resolution_agent.resolve_issue(user_issue)
            st.subheader("Resolution Agent Response:")
            st.write(response)
        else:
            response = escalation_agent.escalate_issue(user_issue)
            st.subheader("Escalation Agent Response:")
            st.write(response)
    else:
        st.warning("Please enter a valid IT issue.")
