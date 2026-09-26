import pandas as pd
import plotly.express as px


def describe_data(df: pd.DataFrame, columns: str = "") -> str:
    """Return summary statistics for numeric columns, or specific columns if given."""
    cols = [c.strip() for c in columns.split(",") if c.strip()] or None
    try:
        subset = df[cols] if cols else df
        return subset.describe(include="all").to_string()
    except KeyError as e:
        return f"Error: column not found — {e}. Available columns: {list(df.columns)}"


def correlation(df: pd.DataFrame, columns: str) -> str:
    """Return the correlation between two or more numeric columns."""
    cols = [c.strip() for c in columns.split(",")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Error: columns not found: {missing}. Available: {list(df.columns)}"
    try:
        return df[cols].corr(numeric_only=True).to_string()
    except Exception as e:
        return f"Error computing correlation: {e}"


def filter_data(df: pd.DataFrame, condition: str) -> str:
    """Filter rows using a pandas query string, e.g. \"region == 'North'\"."""
    try:
        filtered = df.query(condition)
        return f"Filtered to {len(filtered)} rows.\n{filtered.head(5).to_string()}"
    except Exception as e:
        return f"Error: invalid filter condition '{condition}' — {e}"


def group_by_agg(df: pd.DataFrame, group_col: str, agg_col: str, agg_func: str = "mean") -> str:
    """Group by one column and aggregate another (mean, sum, count, min, max, median)."""
    valid_funcs = {"mean", "sum", "count", "min", "max", "median"}
    if agg_func not in valid_funcs:
        return f"Error: agg_func must be one of {valid_funcs}"
    if group_col not in df.columns or agg_col not in df.columns:
        return f"Error: columns not found. Available: {list(df.columns)}"
    try:
        result = df.groupby(group_col)[agg_col].agg(agg_func)
        return result.to_string()
    except Exception as e:
        return f"Error in groupby: {e}"


def plot_column(df: pd.DataFrame, column: str, chart_type: str = "histogram"):
    """Create and return a plotly figure. chart_type: histogram, bar, box, line, scatter."""
    try:
        if chart_type == "scatter":
            x, y = [c.strip() for c in column.split(",")]
            fig = px.scatter(df, x=x, y=y)
        elif chart_type == "histogram":
            fig = px.histogram(df, x=column)
        elif chart_type == "bar":
            fig = px.bar(df, x=column)
        elif chart_type == "box":
            fig = px.box(df, y=column)
        elif chart_type == "line":
            fig = px.line(df, y=column)
        else:
            return f"Error: unknown chart_type '{chart_type}'"
        return fig
    except Exception as e:
        return f"Error creating chart: {e}"