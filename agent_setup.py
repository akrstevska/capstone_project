from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_ollama import OllamaLLM

from tools import get_logs, filter_critical, summarize_logs
from langchain.prompts import PromptTemplate

llm = OllamaLLM(model="llama3")

tools = [
    Tool(
        name="GetRecentLogs",
        func=get_logs,
        description="Gets system logs from the last N minutes. Input should be a number (e.g., '5' for 5 minutes)."
    ),
    Tool(
        name="FilterCriticalLogs",
        func=filter_critical,
        description="Filters logs with severity level 4 or lower (e.g., critical errors). Input should be the result from GetRecentLogs."
    ),
    Tool(
        name="SummarizeLogs",
        func=summarize_logs,
        description="Summarizes a list of logs into a concise report (top 5 lines). Input should be the result from FilterCriticalLogs."
    )
]

prompt_template = PromptTemplate.from_template("""
You are a helpful system log analysis assistant.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question
Thought: your reasoning about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: result of the action
... (this Thought/Action/Action Input/Observation can repeat multiple times)
Thought: I now know the final answer
Final Answer: your final answer to the original question

To analyze logs, you should:
1. First use GetRecentLogs to get recent logs
2. Then use FilterCriticalLogs on the result to find critical errors
3. Finally use SummarizeLogs on the filtered result to get a readable summary

Begin!

Question: {input}
{agent_scratchpad}
""")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={"prompt": prompt_template}
)