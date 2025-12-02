import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import streamlit as st
from dotenv import load_dotenv

import openai  # openai==1.64.0

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


# ============================================================
# Azure OpenAI CONFIG (your credentials)
# ============================================================
load_dotenv()

AZURE_OPENAI_API_KEY = "2ABecnfxzhRg4M5D6pBKiqxXVhmGB2WvQ0aYKkbTCPsj0JLKsZPfJQQJ99BDAC77bzfXJ3w3AAABACOGi3sC"
AZURE_OPENAI_ENDPOINT = (
    "https://openai-api-management-gw.azure-api.net"  # base endpoint
)
AZURE_OPENAI_API_VERSION = "2023-12-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # deployment name

# Azure client (for our helper calls like summarization)
client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# AutoGen config_list compatible with pyautogen==0.7.6
LLM_CONFIG = {
    "config_list": [
        {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "api_key": AZURE_OPENAI_API_KEY,
            "base_url": AZURE_OPENAI_ENDPOINT,
            "api_type": "azure",
            "api_version": AZURE_OPENAI_API_VERSION,
        }
    ],
    "temperature": 0.2,
}

# ============================================================
# Helper: direct LLM call for summarization, etc.
# ============================================================
def call_llm(system_msg: str, user_msg: str) -> str:
    """Helper to call Azure ChatCompletion directly."""
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content


# ============================================================
# Memory Implementations
# ============================================================

class SummarizationMemory:
    """Distills long dialogue into a short summary using the LLM."""

    def summarize(self, dialogue_text: str) -> str:
        if not dialogue_text.strip():
            return "No dialogue yet to summarize."
        system = (
            "You are a memory agent. Given a long conversation, produce a short "
            "summary capturing key decisions, facts, and open questions."
        )
        return call_llm(system, dialogue_text)


@dataclass
class VectorStoreItem:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: set = field(default_factory=set)


class VectorStoreMemory:
    """
    A simple in-memory 'vector store':
    - Tokenizes text into word sets.
    - Uses cosine-like similarity on bag-of-words.
    This is a didactic toy example, not a production vector DB.
    """

    def __init__(self) -> None:
        self.items: List[VectorStoreItem] = []

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(
            w.lower()
            for w in text.replace("\n", " ").split()
            if w.isascii() and w.isalpha()
        )

    @staticmethod
    def _similarity(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / math.sqrt(len(a) * len(b))

    def add(self, text: str, metadata: Dict[str, Any]) -> None:
        tokens = self._tokenize(text)
        self.items.append(VectorStoreItem(text=text, metadata=metadata, tokens=tokens))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, VectorStoreItem]]:
        q_tokens = self._tokenize(query)
        scores = [
            (self._similarity(q_tokens, item.tokens), item) for item in self.items
        ]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s for s in scores[:top_k] if s[0] > 0]


# ============================================================
# Safety Mechanisms
# ============================================================

BANNED_KEYWORDS = [
    "hack",
    "sql injection",
    "malware",
    "delete database",
    "bomb",
    "terrorist",
]


def safety_check(user_input: str) -> Tuple[bool, str]:
    """
    Simple safety gate:
    - Blocks obviously dangerous instructions by keyword.
    - Returns (is_safe, message).
    """
    lower = user_input.lower()
    for kw in BANNED_KEYWORDS:
        if kw in lower:
            return (
                False,
                f"Safety Gate: The request contains a blocked keyword: '{kw}'. "
                "This workflow will not run.",
            )
    return True, "Input passed safety check."


# ============================================================
# Build Agent Teams (Planning vs Reactive)
# ============================================================

def build_agents(planning_mode: str, max_round: int = 12):
    """
    planning_mode: "Planning" or "Reactive"
    Returns (user_proxy, manager)
    """

    # User proxy (represents the human)
    user_proxy = UserProxyAgent(
        name="HumanUser",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config=False,
        system_message=(
            "You represent the human user. You pass their request into the group "
            "and show the final answer back to them."
        ),
    )

    # Researcher
    researcher = AssistantAgent(
        name="Researcher",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a research assistant.\n"
            "- Carefully read the task.\n"
            "- If a plan is provided, follow it.\n"
            "- Produce a structured answer with sections:\n"
            "  OVERVIEW, KEY POINTS, BENEFITS / USE-CASES, RISKS / LIMITATIONS, EXAMPLES.\n"
        ),
        description="Main worker that does in-depth reasoning & drafting.",
    )

    # Reviewer
    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=LLM_CONFIG,
        system_message=(
            "You are a strict reviewer.\n"
            "- Read the Researcher's answer (and Planner's plan if present).\n"
            "- Fix clarity, remove redundancy, and ensure consistency.\n"
            "- Optimize it for a training / teaching context.\n"
            "- End with a line: 'FINAL_DECISION: APPROVED'."
        ),
        description="Reviews and polishes the answer, acts as final gate.",
    )

    agents = [user_proxy, researcher, reviewer]

    # Optional Planner
    if planning_mode == "Planning":
        planner = AssistantAgent(
            name="Planner",
            llm_config=LLM_CONFIG,
            system_message=(
                "You are a planning agent.\n"
                "- For each user task, first produce a numbered high-level plan.\n"
                "- Then give clear instructions to the Researcher and Reviewer "
                "about how to execute the plan.\n"
                "- Your messages should contain a 'PLAN:' section."
            ),
            description="Creates a plan before work starts.",
        )
        agents.insert(1, planner)  # order: HumanUser, Planner, Researcher, Reviewer

    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,           # timeout-like control
        speaker_selection_method="auto",
        send_introductions=True,
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=LLM_CONFIG,
    )

    return user_proxy, manager


# ============================================================
# Run Multi-Agent Workflow
# ============================================================

def run_workflow(
    user_topic: str,
    extra_instruction: str,
    planning_mode: str,
    memory_mode: str,
    vec_mem: VectorStoreMemory,
    sum_mem: SummarizationMemory,
) -> Dict[str, Any]:
    """
    Run one multi-agent workflow and update memory if needed.
    Returns dict with:
        final_answer, messages, summary_memory, vector_results
    """

    # Safety Gate
    is_safe, safety_msg = safety_check(user_topic + "\n" + extra_instruction)
    if not is_safe:
        return {
            "final_answer": safety_msg,
            "messages": [],
            "summary_memory": "",
            "vector_results": [],
        }

    user_proxy, manager = build_agents(planning_mode=planning_mode, max_round=12)

    workflow_desc = (
        "Workflow:\n"
        "1. If a Planner is present, Planner first creates a PLAN.\n"
        "2. Researcher executes the plan / directly answers for the user.\n"
        "3. Reviewer refines and approves the final answer.\n"
        "Reviewer must end with 'FINAL_DECISION: APPROVED'."
    )

    message = f"""
User topic / question:
{user_topic}

Additional instructions:
{extra_instruction}

{workflow_desc}

Begin.
"""

    chat_result = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    messages = manager.groupchat.messages

    # Extract final answer (prefer Reviewer if present)
    final_answer = "No final answer produced."
    for m in reversed(messages):
        if m.get("name") == "Reviewer":
            final_answer = m.get("content", "")
            break
    else:
        if messages:
            final_answer = messages[-1].get("content", "")

    # ---- Memory handling ----
    dialogue_text = "\n\n".join(
        f"{m.get('name','')}:\n{m.get('content','')}" for m in messages
    )

    summary_memory = ""
    vector_results: List[Tuple[float, VectorStoreItem]] = []

    if memory_mode == "Summarization":
        summary_memory = sum_mem.summarize(dialogue_text)

    elif memory_mode == "Vector Store":
        # store current conversation
        vec_mem.add(
            text=final_answer,
            metadata={"topic": user_topic},
        )
        vector_results = vec_mem.search(user_topic, top_k=3)

    return {
        "final_answer": final_answer,
        "messages": messages,
        "summary_memory": summary_memory,
        "vector_results": vector_results,
    }


# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="Agent Memory, Planning & Safety (AutoGen)",
        layout="wide",
    )
    st.title("🧠 Agent Memory, Planning vs Reactive, Multi-Agent & Safety Demo")

    st.markdown(
        """
This app demonstrates **four key concepts**:

1. **Agent Memory Strategies**
   - *Summarization memory*: distill long chats into compact summaries.
   - *Vector-store memory*: store & retrieve past conversations.

2. **Planning vs. Reactive Agents**
   - **Reactive mode**: Researcher + Reviewer only.
   - **Planning mode**: Planner → Researcher → Reviewer pipeline.

3. **Multi-Agent Collaboration Patterns**
   - Multiple agents (Planner, Researcher, Reviewer) collaborating
     in a coordinated workflow.

4. **Safety in Agent Systems**
   - Simple safety gate (banned keywords).
   - Max rounds acting like a timeout control.
"""
    )

    # ----- Sidebar controls -----
    st.sidebar.header("⚙️ Configuration")

    planning_mode = st.sidebar.radio(
        "Agent Style",
        options=["Reactive", "Planning"],
        help="Reactive: Researcher & Reviewer.\nPlanning: Planner + Researcher + Reviewer.",
    )

    memory_mode = st.sidebar.radio(
        "Memory Strategy",
        options=["None", "Summarization", "Vector Store"],
        help="Choose how the system maintains memory across conversations.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Safety Gate:**\n\n"
        "- Blocks obvious dangerous keywords.\n"
        "- Max rounds in group chat acts like a timeout."
    )

    # ----- Session-state memory -----
    if "vector_memory" not in st.session_state:
        st.session_state["vector_memory"] = VectorStoreMemory()
    if "summarization_memory" not in st.session_state:
        st.session_state["summarization_memory"] = SummarizationMemory()

    vec_mem: VectorStoreMemory = st.session_state["vector_memory"]
    sum_mem: SummarizationMemory = st.session_state["summarization_memory"]

    # ----- Main input -----
    default_topic = "Explain agent memory strategies (summarization vs vector-store) and when to use them in agentic AI systems."
    user_topic = st.text_area(
        "💬 Enter a topic / question for the agents:",
        value=default_topic,
        height=130,
    )

    extra_instruction = st.text_area(
        "✏️ Extra instructions (optional):",
        value="Explain in a way I can directly use in a training session, with bullet points and simple examples.",
        height=90,
    )

    if st.button("🚀 Run Multi-Agent Workflow"):
        if not user_topic.strip():
            st.error("Please enter a topic or question.")
            return

        with st.spinner("Agents are collaborating..."):
            result = run_workflow(
                user_topic=user_topic,
                extra_instruction=extra_instruction,
                planning_mode=planning_mode,
                memory_mode=memory_mode,
                vec_mem=vec_mem,
                sum_mem=sum_mem,
            )

        final_answer = result["final_answer"]
        messages = result["messages"]
        summary_memory = result["summary_memory"]
        vector_results = result["vector_results"]

        st.success("Workflow completed.")

        # ----- Final Answer -----
        st.subheader("✅ Final Reviewed Answer")
        st.write(final_answer)

        # ----- Memory Views -----
        if memory_mode == "Summarization":
            st.subheader("🧠 Summarization Memory (Distilled Dialogue)")
            st.write(summary_memory)

        elif memory_mode == "Vector Store":
            st.subheader("🧠 Vector-Store Memory: Retrieved Similar Past Answers")
            if not vector_results:
                st.info("No similar past conversations yet (or similarity too low).")
            else:
                for score, item in vector_results:
                    st.markdown(f"**Similarity score:** {score:.3f}")
                    st.markdown(f"**Topic:** {item.metadata.get('topic', 'N/A')}")
                    st.write(item.text)
                    st.markdown("---")

        # ----- Conversation Transcript -----
        with st.expander("📝 Full Conversation Transcript (All Agents)"):
            if not messages:
                st.write("No messages (blocked by safety gate?).")
            else:
                for msg in messages:
                    name = msg.get("name", "Agent")
                    content = msg.get("content", "")
                    st.markdown(f"### {name}")
                    st.write(content)
                    st.markdown("---")

    # Helper explanation section
    st.markdown("---")
    st.markdown(
        """
### How this maps to the syllabus line

- **Agent memory strategies**  
  - *Summarization memory* → choose **Summarization** in the sidebar.  
  - *Vector-store memory* → choose **Vector Store**; previous runs are stored & retrieved.

- **Planning vs. reactive agents**  
  - *Reactive* → **Reactive** mode (Researcher + Reviewer).  
  - *Planning* → **Planning** mode (Planner + Researcher + Reviewer).

- **Multi-agent systems / collaboration patterns**  
  - All modes use multiple agents; Planning adds an explicit Planner stage.

- **Safety in agent systems**  
  - **Safety gate** blocks dangerous inputs (banned keywords).  
  - **max_round** in `GroupChat` acts like a timeout control for runaway loops.
"""
    )


if __name__ == "__main__":
    main()
