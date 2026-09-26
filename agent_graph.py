import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
import pandas as pd

load_dotenv()

# ---- Tools ----

@tool
def describe_data(columns: str = "") -> str:
    """Return summary statistics for numeric columns in the dataset."""
    df = pd.read_csv("data/sales.csv")
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    subset = df[cols] if cols else df
    return subset.describe(include="all").to_string()

@tool
def correlation(columns: str) -> str:
    """Return correlation between comma-separated numeric columns."""
    df = pd.read_csv("data/sales.csv")
    cols = [c.strip() for c in columns.split(",")]
    return df[cols].corr(numeric_only=True).to_string()

@tool
def filter_data(condition: str) -> str:
    """Filter rows using a pandas query string."""
    df = pd.read_csv("data/sales.csv")
    filtered = df.query(condition)
    return f"Filtered to {len(filtered)} rows.\n{filtered.head(5).to_string()}"

@tool
def group_by_agg(group_col: str, agg_col: str, agg_func: str = "mean") -> str:
    """Group by one column and aggregate another (mean, sum, count, min, max, median)."""
    df = pd.read_csv("data/sales.csv")
    result = df.groupby(group_col)[agg_col].agg(agg_func)
    return result.to_string()

@tool
def sort_data(column: str, ascending: bool = False, top_n: int = 5) -> str:
    """Sort rows by a column and return the top N rows.
    Set ascending=True for lowest-first, False (default) for highest-first."""
    df = pd.read_csv("data/sales.csv")
    if column not in df.columns:
        return f"Error: column '{column}' not found. Available: {list(df.columns)}"
    sorted_df = df.sort_values(by=column, ascending=ascending).head(top_n)
    return sorted_df.to_string()

@tool
def value_counts(column: str) -> str:
    """Count how many times each unique value appears in a column.
    Useful for questions like 'how many sales per region' or 'most common product'."""
    df = pd.read_csv("data/sales.csv")
    if column not in df.columns:
        return f"Error: column '{column}' not found. Available: {list(df.columns)}"
    counts = df[column].value_counts()
    return counts.to_string()

@tool
def missing_data_check() -> str:
    """Check for missing (null/blank) values in each column of the dataset."""
    df = pd.read_csv("data/sales.csv")
    missing = df.isnull().sum()
    total_rows = len(df)
    if missing.sum() == 0:
        return f"No missing values found. Dataset has {total_rows} rows and {len(df.columns)} columns, all complete."
    result = missing[missing > 0]
    return f"Missing values found (out of {total_rows} rows):\n{result.to_string()}"

tools = [
    describe_data,
    correlation,
    filter_data,
    group_by_agg,
    sort_data,
    value_counts,
    missing_data_check,
]
tool_map = {t.name: t for t in tools}

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ---- State ----

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: str
    retries: int
    steps: int

# ---- Node 0: plan what to do before acting ----

def plan_node(state: AgentState) -> dict:
    print("\n[NODE: plan] thinking about approach...")
    planning_prompt = SystemMessage(content=(
        "You are a data analysis planner. Based on the conversation so far, "
        "state in ONE short sentence what the next analysis step should be. "
        "Do not call any tools here — just describe the plan in plain text."
    ))
    try:
        response = llm.invoke([planning_prompt] + state["messages"])
        plan_text = response.content
    except Exception as e:
        print(f"[NODE: plan] planning failed ({e}), using fallback plan.")
        plan_text = "Continue analyzing based on the most recent tool result."
    print(f"[NODE: plan] plan: {plan_text}")
    return {"plan": plan_text}

# ---- Node 1: ask the LLM to decide on a tool ----

def call_llm_node(state: AgentState) -> dict:
    print("\n[NODE: call_llm] deciding on a tool...")
    plan_reminder = SystemMessage(content=f"Current plan: {state.get('plan', '')}")
    response = llm_with_tools.invoke([plan_reminder] + state["messages"])
    return {"messages": [response]}

# ---- Node 2: run whatever tool the LLM asked for ----

def run_tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_messages = []
    had_error = False
    for call in last_msg.tool_calls:
        print(f"[NODE: run_tools] calling {call['name']}({call['args']})")
        tool_fn = tool_map.get(call["name"])
        if tool_fn is None:
            result = f"Error: unknown tool '{call['name']}'"
            had_error = True
        else:
            try:
                result = tool_fn.invoke(call["args"])
            except Exception as e:
                result = f"Error running tool: {e}. Try different arguments or a different tool."
                had_error = True
                print(f"[NODE: run_tools] ERROR: {e}")
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    updates = {
        "messages": tool_messages,
        "steps": state.get("steps", 0) + 1,
    }
    if had_error:
        updates["retries"] = state.get("retries", 0) + 1
    return updates

# ---- Node 3: give up gracefully after too many errors or too many steps ----

def give_up_node(state: AgentState) -> dict:
    print("[NODE: give_up] stopping — too many retries or steps.")
    return {"messages": [AIMessage(content=(
        "I wasn't able to fully answer this question — either the data doesn't contain "
        "what was asked for, or I ran into repeated errors. Try rephrasing your question "
        "or checking the column names in your file."
    ))]}

# ---- Decision: does the LLM want a tool, or is it done? ----

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "run_tools"
    return "end"

# ---- Decision: after running tools, retry, give up, or keep going ----

MAX_RETRIES = 3
MAX_STEPS = 6

def check_retry_limit(state: AgentState) -> str:
    if state.get("retries", 0) >= MAX_RETRIES:
        return "give_up"
    if state.get("steps", 0) >= MAX_STEPS:
        return "give_up"
    return "plan"

# ---- Build the graph ----

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("call_llm", call_llm_node)
graph.add_node("run_tools", run_tools_node)
graph.add_node("give_up", give_up_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "call_llm")
graph.add_conditional_edges("call_llm", should_continue, {
    "run_tools": "run_tools",
    "end": END,
})
graph.add_conditional_edges("run_tools", check_retry_limit, {
    "plan": "plan",
    "give_up": "give_up",
})
graph.add_edge("give_up", END)

app_graph = graph.compile()

# ---- Test it ----

question = "Is there any missing data in this dataset?"
initial_state = {
    "messages": [HumanMessage(content=question)],
    "plan": "",
    "retries": 0,
    "steps": 0,
}

final_state = app_graph.invoke(initial_state, config={"recursion_limit": 25})

print("\n---- FINAL ANSWER ----")
print(final_state["messages"][-1].content)