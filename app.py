import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #1a2035 0%, #0F1419 60%);
    }
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(255,107,53,0.25);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0.3rem 0 0 0;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 4px;
        background-color: rgba(255,255,255,0.03);
        margin-bottom: 6px;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 10px;
    }
    .stButton button {
        border-radius: 20px;
        border: 1px solid #FF6B35;
        background-color: rgba(255,107,53,0.08);
        color: #FF6B35;
        transition: all 0.2s ease;
        font-size: 0.85rem;
    }
    .stButton button:hover {
        background-color: #FF6B35;
        color: white;
        border-color: #FF6B35;
    }
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid #2D333B;
        background-color: rgba(255,255,255,0.02);
    }
    [data-testid="stChatInput"] {
        border-radius: 14px;
        border: 1.5px solid #FF6B35 !important;
    }
    .suggestion-label {
        color: #888;
        font-size: 0.8rem;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ---- Shared holders ----
current_df = {"data": None}
current_fig = {"fig": None}

# ---- Tools ----

@tool
def describe_data(columns: str = "") -> str:
    """Return summary statistics for numeric columns in the dataset."""
    df = current_df["data"]
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    subset = df[cols] if cols else df
    return subset.describe(include="all").to_string()

@tool
def correlation(columns: str) -> str:
    """Return correlation between comma-separated numeric columns."""
    df = current_df["data"]
    cols = [c.strip() for c in columns.split(",")]
    return df[cols].corr(numeric_only=True).to_string()

@tool
def filter_data(condition: str) -> str:
    """Filter rows using a pandas query string."""
    df = current_df["data"]
    filtered = df.query(condition)
    return f"Filtered to {len(filtered)} rows.\n{filtered.head(5).to_string()}"

@tool
def group_by_agg(group_col: str, agg_col: str, agg_func: str = "mean") -> str:
    """Group by one column and aggregate another (mean, sum, count, min, max, median)."""
    df = current_df["data"]
    result = df.groupby(group_col)[agg_col].agg(agg_func)
    return result.to_string()

@tool
def sort_data(column: str, ascending: bool = False, top_n: int = 5) -> str:
    """Sort rows by a column and return the top N rows."""
    df = current_df["data"]
    if column not in df.columns:
        return f"Error: column '{column}' not found. Available: {list(df.columns)}"
    sorted_df = df.sort_values(by=column, ascending=ascending).head(top_n)
    return sorted_df.to_string()

@tool
def value_counts(column: str) -> str:
    """Count how many times each unique value appears in a column."""
    df = current_df["data"]
    if column not in df.columns:
        return f"Error: column '{column}' not found. Available: {list(df.columns)}"
    counts = df[column].value_counts()
    return counts.to_string()

@tool
def missing_data_check() -> str:
    """Check for missing (null/blank) values in each column of the dataset."""
    df = current_df["data"]
    missing = df.isnull().sum()
    total_rows = len(df)
    if missing.sum() == 0:
        return f"No missing values found. Dataset has {total_rows} rows and {len(df.columns)} columns, all complete."
    result = missing[missing > 0]
    return f"Missing values found (out of {total_rows} rows):\n{result.to_string()}"

@tool
def plot_column(column: str, chart_type: str = "histogram") -> str:
    """Create a chart. chart_type: histogram, bar, box, line, scatter.
    For scatter, pass column as 'x_col,y_col'."""
    df = current_df["data"]
    try:
        if chart_type == "scatter":
            x, y = [c.strip() for c in column.split(",")]
            fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        elif chart_type == "bar":
            fig = px.bar(df, x=column, title=f"Bar chart of {column}")
        elif chart_type == "box":
            fig = px.box(df, y=column, title=f"Box plot of {column}")
        elif chart_type == "line":
            fig = px.line(df, y=column, title=f"Trend of {column}")
        else:
            return f"Error: unknown chart_type '{chart_type}'"
        current_fig["fig"] = fig
        return f"Chart created successfully: {chart_type} of {column}. It will be displayed to the user."
    except Exception as e:
        return f"Error creating chart: {e}"

@tool
def detect_outliers(column: str) -> str:
    """Detect outliers in a numeric column using the IQR (interquartile range) method.
    Returns how many outliers were found and their values."""
    df = current_df["data"]
    if column not in df.columns:
        return f"Error: column '{column}' not found. Available: {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Error: '{column}' is not a numeric column."

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    if outliers.empty:
        return f"No outliers found in '{column}'. Normal range is roughly {lower_bound:.2f} to {upper_bound:.2f}."

    return (
        f"Found {len(outliers)} outlier(s) in '{column}' "
        f"(normal range: {lower_bound:.2f} to {upper_bound:.2f}):\n"
        f"{outliers.to_string()}"
    )

@tool
def generate_insights() -> str:
    """Scan the entire dataset and proactively surface a few interesting findings,
    such as notable differences between categories or strong correlations.
    Use this when the user asks for a general overview, summary, or 'tell me something interesting'."""
    df = current_df["data"]
    insights = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        corr_pairs = (
            corr_matrix.where(~corr_matrix.isna())
            .unstack()
            .sort_values(ascending=False)
        )
        corr_pairs = corr_pairs[corr_pairs < 0.999]
        if not corr_pairs.empty:
            top_pair = corr_pairs.index[0]
            top_value = corr_pairs.iloc[0]
            insights.append(
                f"Strongest relationship: '{top_pair[0]}' and '{top_pair[1]}' "
                f"have a correlation of {top_value:.2f}."
            )

    for cat_col in categorical_cols[:2]:
        for num_col in numeric_cols[:2]:
            grouped = df.groupby(cat_col)[num_col].mean()
            if len(grouped) >= 2:
                highest = grouped.idxmax()
                lowest = grouped.idxmin()
                gap_pct = ((grouped.max() - grouped.min()) / grouped.min()) * 100 if grouped.min() != 0 else 0
                insights.append(
                    f"'{highest}' has the highest average {num_col} ({grouped.max():.2f}) among {cat_col}s, "
                    f"while '{lowest}' has the lowest ({grouped.min():.2f}) — a {gap_pct:.0f}% difference."
                )

    missing_total = df.isnull().sum().sum()
    insights.append(
        f"Dataset has {len(df)} rows and {len(df.columns)} columns, "
        f"with {missing_total} missing value(s) total."
    )

    if not insights:
        return "Not enough numeric/categorical structure in this dataset to generate automatic insights."

    return "\n".join(f"- {i}" for i in insights)

tools = [
    describe_data, correlation, filter_data, group_by_agg, sort_data,
    value_counts, missing_data_check, plot_column, detect_outliers, generate_insights,
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
    trace: list

def plan_node(state: AgentState) -> dict:
    planning_prompt = SystemMessage(content=(
        "You are a data analysis planner. Based on the conversation so far, "
        "state in ONE short sentence what the next analysis step should be. "
        "Do not call any tools here — just describe the plan in plain text."
    ))
    try:
        response = llm.invoke([planning_prompt] + state["messages"])
        plan_text = response.content
    except Exception:
        plan_text = "Continue analyzing based on the most recent tool result."
    trace = state.get("trace", []) + [f"🧭 **Plan:** {plan_text}"]
    return {"plan": plan_text, "trace": trace}

def call_llm_node(state: AgentState) -> dict:
    plan_reminder = SystemMessage(content=f"Current plan: {state.get('plan', '')}")
    response = llm_with_tools.invoke([plan_reminder] + state["messages"])
    trace = state.get("trace", [])
    if response.tool_calls:
        for call in response.tool_calls:
            trace = trace + [f"🔧 **Calling tool:** `{call['name']}({call['args']})`"]
    return {"messages": [response], "trace": trace}

def run_tools_node(state: AgentState) -> dict:
    last_msg = state["messages"][-1]
    tool_messages = []
    had_error = False
    trace = state.get("trace", [])
    for call in last_msg.tool_calls:
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
        trace = trace + [f"📊 **Result:** {str(result)[:300]}"]
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    updates = {"messages": tool_messages, "steps": state.get("steps", 0) + 1, "trace": trace}
    if had_error:
        updates["retries"] = state.get("retries", 0) + 1
    return updates

def give_up_node(state: AgentState) -> dict:
    return {"messages": [AIMessage(content=(
        "I wasn't able to fully answer this question — either the data doesn't contain "
        "what was asked for, or I ran into repeated errors. Try rephrasing your question "
        "or checking the column names in your file."
    ))]}

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "run_tools"
    return "end"

MAX_RETRIES = 3
MAX_STEPS = 6

def check_retry_limit(state: AgentState) -> str:
    if state.get("retries", 0) >= MAX_RETRIES:
        return "give_up"
    if state.get("steps", 0) >= MAX_STEPS:
        return "give_up"
    return "plan"

graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("call_llm", call_llm_node)
graph.add_node("run_tools", run_tools_node)
graph.add_node("give_up", give_up_node)
graph.set_entry_point("plan")
graph.add_edge("plan", "call_llm")
graph.add_conditional_edges("call_llm", should_continue, {"run_tools": "run_tools", "end": END})
graph.add_conditional_edges("run_tools", check_retry_limit, {"plan": "plan", "give_up": "give_up"})
graph.add_edge("give_up", END)
app_graph = graph.compile()

def run_agent(question: str) -> tuple[str, list, object]:
    """Runs the agent graph for one question. Returns (answer, trace, figure)."""
    current_fig["fig"] = None
    df = current_df["data"]

    context_message = SystemMessage(content=(
        f"A dataset is already loaded and available to your tools — you do NOT need to ask "
        f"the user to upload it or describe it. "
        f"It has {len(df)} rows and these columns: {list(df.columns)}, "
        f"with these data types: {df.dtypes.astype(str).to_dict()}. "
        f"Use the available tools directly to answer the user's question about this data. "
        f"IMPORTANT: Only state facts, numbers, and values that come directly from a tool's output. "
        f"Never invent, estimate, or assume any value, category, or number that wasn't explicitly "
        f"returned by a tool call. If you're unsure, call another tool to check rather than guessing. "
        f"When formatting your answer, use plain Markdown only (bullet points, bold text, tables) — "
        f"never use raw HTML tags like <br> or <div>."
    ))

    initial_state = {
        "messages": [context_message, HumanMessage(content=question)],
        "plan": "",
        "retries": 0,
        "steps": 0,
        "trace": [],
    }
    final_state = app_graph.invoke(initial_state, config={"recursion_limit": 25})
    answer = final_state["messages"][-1].content
    trace = final_state.get("trace", [])
    fig = current_fig["fig"]
    return answer, trace, fig

# ============ STREAMLIT UI ============

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Sidebar: upload + dataset info ----
with st.sidebar:
    st.header("📁 Dataset")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")

    clicked_example = None

    if uploaded_file is not None:
        current_df["data"] = pd.read_csv(uploaded_file)
        df = current_df["data"]

        st.success("File loaded successfully")

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF6B35 0%, #FF6B3520 100%);
                    border-left: 4px solid #FF6B35; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
            <span style="color: #aaa; font-size: 0.8rem;">ROWS</span><br>
            <span style="color: white; font-size: 1.6rem; font-weight: 700;">{len(df)}</span>
        </div>
        <div style="background: linear-gradient(135deg, #00C2CB 0%, #00C2CB20 100%);
                    border-left: 4px solid #00C2CB; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
            <span style="color: #aaa; font-size: 0.8rem;">COLUMNS</span><br>
            <span style="color: white; font-size: 1.6rem; font-weight: 700;">{len(df.columns)}</span>
        </div>
        <div style="background: linear-gradient(135deg, #8B5CF6 0%, #8B5CF620 100%);
                    border-left: 4px solid #8B5CF6; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
            <span style="color: #aaa; font-size: 0.8rem;">MISSING VALUES</span><br>
            <span style="color: white; font-size: 1.6rem; font-weight: 700;">{int(df.isnull().sum().sum())}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Column names & types"):
            st.dataframe(df.dtypes.astype(str).rename("type"))

        st.divider()
        if st.session_state.chat_history:
            summary_lines = [
                "# AI Data Analyst — Session Summary",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
            ]
            for role, content, _ in st.session_state.chat_history:
                label = "**You:**" if role == "user" else "**Assistant:**"
                summary_lines.append(f"{label} {content}\n")
            summary_text = "\n".join(summary_lines)
            st.download_button(
                "⬇️ Export conversation",
                data=summary_text,
                file_name="data_analysis_summary.md",
                mime="text/markdown",
                use_container_width=True,
            )

# ---- Main area: gradient header ----
st.markdown("""
<div class="main-header">
    <h1>📊 AI Data Analyst</h1>
    <p>Upload a CSV in the sidebar, then ask questions about it in plain English.</p>
</div>
""", unsafe_allow_html=True)

if uploaded_file is not None:
    for role, content, fig in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(content)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    # ---- Suggested question chips, right above the input ----
    st.markdown('<div class="suggestion-label">💡 Suggested questions</div>', unsafe_allow_html=True)
    example_questions = [
        "📋 Give me an overview",
        "🏆 Highest average price by region",
        "🎯 Any outliers in price?",
        "📊 Show histogram of price",
        "🧹 Any missing data?",
    ]
    question_map = {
        "📋 Give me an overview": "Give me an overview of this data",
        "🏆 Highest average price by region": "Which region has the highest average price?",
        "🎯 Any outliers in price?": "Are there any outliers in price?",
        "📊 Show histogram of price": "Show me a histogram of price",
        "🧹 Any missing data?": "Is there any missing data?",
    }
    chip_cols = st.columns(len(example_questions))
    for i, label in enumerate(example_questions):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                clicked_example = question_map[label]

    typed_question = st.chat_input("Ask a question about your data...")
    question = clicked_example or typed_question

    if question:
        st.session_state.chat_history.append(("user", question, None))
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            answer, trace, fig = run_agent(question)

        with st.chat_message("assistant"):
            st.write(answer)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            with st.expander("🔍 See how I got this answer"):
                for step in trace:
                    st.markdown(step)

        st.session_state.chat_history.append(("assistant", answer, fig))
        st.rerun()
else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.03);
                border-radius: 16px; border: 1px dashed #FF6B35;">
        <div style="font-size: 3rem;">📁</div>
        <h3 style="color: white;">No dataset loaded yet</h3>
        <p style="color: #aaa;">Upload a CSV file using the sidebar to start exploring your data with AI.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8rem;">📈</div>
            <b style="color: white;">Analyze trends</b><br>
            <span style="color: #aaa; font-size: 0.85rem;">Group, sort, and compare your data</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8rem;">🎯</div>
            <b style="color: white;">Spot outliers</b><br>
            <span style="color: #aaa; font-size: 0.85rem;">Automatically flag unusual values</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border-radius: 10px; padding: 16px; text-align: center;">
            <div style="font-size: 1.8rem;">💬</div>
            <b style="color: white;">Just ask</b><br>
            <span style="color: #aaa; font-size: 0.85rem;">Plain English questions, real answers</span>
        </div>
        """, unsafe_allow_html=True)