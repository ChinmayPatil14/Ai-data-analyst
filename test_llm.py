import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import pandas as pd

load_dotenv()

@tool
def describe_data(columns: str = "") -> str:
    """Return summary statistics for numeric columns in the dataset.
    Pass a comma-separated column list, or empty string for all columns."""
    df = pd.read_csv("data/sales.csv")
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    subset = df[cols] if cols else df
    return subset.describe(include="all").to_string()

@tool
def correlation(columns: str) -> str:
    """Return the correlation between two or more numeric columns.
    Pass comma-separated column names, e.g. 'price,quantity'."""
    df = pd.read_csv("data/sales.csv")
    cols = [c.strip() for c in columns.split(",")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Error: columns not found: {missing}. Available: {list(df.columns)}"
    return df[cols].corr(numeric_only=True).to_string()

@tool
def filter_data(condition: str) -> str:
    """Filter rows using a pandas query string, e.g. "region == 'North'"."""
    df = pd.read_csv("data/sales.csv")
    try:
        filtered = df.query(condition)
        return f"Filtered to {len(filtered)} rows.\n{filtered.head(5).to_string()}"
    except Exception as e:
        return f"Error: invalid filter condition '{condition}' — {e}"

@tool
def group_by_agg(group_col: str, agg_col: str, agg_func: str = "mean") -> str:
    """Group by one column and aggregate another.
    agg_func: mean, sum, count, min, max, median."""
    df = pd.read_csv("data/sales.csv")
    if group_col not in df.columns or agg_col not in df.columns:
        return f"Error: columns not found. Available: {list(df.columns)}"
    result = df.groupby(group_col)[agg_col].agg(agg_func)
    return result.to_string()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

tools = [describe_data, correlation, filter_data, group_by_agg]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ---- try a question that needs group_by_agg, not describe_data ----
def run_agent(question, tools, tool_map, llm_with_tools, max_steps=5):
    messages = [HumanMessage(content=question)]

    for step in range(max_steps):
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        if not ai_response.tool_calls:
            # No more tools needed — this is the final answer
            return ai_response.content

        print(f"\n[Step {step+1}] LLM wants to call:")
        for call in ai_response.tool_calls:
            print(f"  {call['name']}({call['args']})")

        for call in ai_response.tool_calls:
            tool_fn = tool_map[call["name"]]
            result = tool_fn.invoke(call["args"])
            print(f"  -> result: {result[:200]}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Gave up after max_steps without a final answer."


# ---- test with a question that might need multiple steps ----
question = "Which region has the highest average price, and how many rows does that region have?"
answer = run_agent(question, tools, tool_map, llm_with_tools)

print("\n---- FINAL ANSWER ----")
print(answer)