import os
import copy
import logging
import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from sqlalchemy import create_engine, text
from langchain_core.tools import StructuredTool
from models import AgentState, SheetSchema
from config import logger, SQLITE_DIR

from worker import TicketAnalysisAgent

_WORKER_INSTANCE = None

def _clean_df_columns(df):
    """Ensures DataFrame columns are valid, non-empty SQLite identifiers."""
    import re
    new_cols = []
    for i, c in enumerate(df.columns):
        # Replace special chars with underscore, strip leading/trailing
        cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', str(c)).strip('_')
        if not cleaned:
            cleaned = f"col_{i}"
        
        # Deduplicate column names
        orig = cleaned
        count = 1
        while cleaned in new_cols:
            cleaned = f"{orig}_{count}"
            count += 1
        new_cols.append(cleaned)
    df.columns = new_cols
    return df

def get_worker()->TicketAnalysisAgent:
    """Retturns a cached instance of the worker to avoid re-initialization overhead"""
    global _WORKER_INSTANCE
    if _WORKER_INSTANCE is None:
        _WORKER_INSTANCE = TicketAnalysisAgent()
    return _WORKER_INSTANCE

def create_sql_tool(engine: Any, target_state: Dict[str, Any]):
    """
    Factory function that creates a SQL tool with the engine AND state bound to it.
    It universally handles SELECT, UPDATE, DELETE, DROP, CREATE, etc.
    by checking the 'returns_rows' property of the result.
    """
    def execute_sql_func(query: str):
        logger.info(f"[SQL AGENT] 🛠️ Executing SQL: {query}")
        
        # 1. Strip cleanup
        query = query.strip().rstrip(';')

        try:
            with engine.begin() as conn:
                result = conn.exec_driver_sql(query)
                
                if result.returns_rows:

                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    row_count, col_count = df.shape

                    if row_count == 1 and col_count == 1:
                        target_state['modified_df'] = None
                    else:
                        target_state['modified_df'] = df
                    
                    logger.info(f"[SQL AGENT] ✅ Query Passed. Rows returned: {row_count}")

                    display_df = df.head(50).copy()
                    display_df.index = np.arange(1, len(display_df) + 1)
                    context_md = display_df.to_markdown(index=True)

                    # 2. Return a unified message with STRICT instructions
                    return (
                        f"Query executed successfully. Result contains {row_count} rows.\n\n"
                        f"SUCCESS_CONFIRMATION: The full data has been sent to the User's UI.\n"
                        f"--- SYSTEM INSTRUCTION ---\n"
                        f"1. The user can ALREADY see the table above. Do NOT output a text table.\n"
                        f"2. You may use the [DATA CONTEXT] below to answer specific questions (e.g., 'What is the status of ID 5?').\n"
                        f"3. If the user just asked to 'show' data, simply say 'I have retrieved the data for you.' and stop.\n"
                        f"--------------------------\n\n"
                        f"[DATA CONTEXT - FOR YOUR EYES ONLY]:\n{context_md}"
                    )

                else:
                    
                    rows_affected = result.rowcount
                    return f"Command executed successfully. Rows affected: {rows_affected}."

        except Exception as e:
            if "You can only execute one statement at a time" in str(e):
                return "Error: SQLite allows only ONE statement at a time. Please split your query into two separate tool calls."
                
            logger.error(f"[SQL AGENT] ❌ Query Failed. Error: {e}")
            return f"SQL Execution Error: {e}"

    return StructuredTool.from_function(
        func=execute_sql_func,
        name="execute_sql_tool",
        description="Executes a SQL query. The tool automatically detects if data is returned or if the database is modified."
    )

# Tool Functions

async def _analyse_sheet_internal(worker, df, sheet_name: str, fpath: str) -> str:
    """Shared logic: Scans a single sheet for columns and unknown ticket types."""
    schema = await worker.find_schema(sheet_name, df)
    
    # Detect unknown ticket types
    unknown_types = await worker.classify_unknown_ticket_types(
        df, 
        id_col=schema.id_column, 
        summary_col=schema.summary_column, 
        file_path=fpath
    )
    
    unknown_types_text = ""
    if unknown_types:
        unknown_types_text = "\n\n⚠️ UNKNOWN TICKET TYPES DETECTED:\n"
        unknown_types_text += "The following ticket ID prefixes were found but are NOT in the standard type list (INC, SR, HC):\n"
        for prefix, info in unknown_types.items():
            suggested = info.get("suggested_type_name", prefix)
            count = info.get("count", 0)
            samples = info.get("sample_ids", [])[:3]
            unknown_types_text += f"- Prefix '{prefix}': {count} tickets (e.g., {samples}) — Suggested type: '{suggested}'\n"
        unknown_types_text += (
            "\nPROTOCOL FOR UNKNOWN TYPES: Present these detected unknown ticket types to the user. "
            "Tell them the suggested classification and ask them to:\n"
            "1. Confirm or correct the ticket type name for each prefix.\n"
            "2. Provide the estimated effort hours per ticket for each new type.\n"
            "Include these in the trigger_analysis call as 'ticket_type_config'."
        )
    
    return (
        f"Sheet '{sheet_name}' analyzed.\n"
        f"- Detected ID Column: {schema.id_column}\n"
        f"- Detected Summary Column: {schema.summary_column}\n"
        f"- Detected Resolution Column: {schema.resolution_column}\n\n"
        f"Known ticket types found: INC (Incident), SR (Service Request), HC (Health Check)"
        f"{unknown_types_text}\n\n"
        "PROTOCOL: Present the detected columns to the user. Ask them to confirm the column mapping "
        "and specify how many months of data this file contains."
    )


async def tool_scan_file(state: AgentState) -> str:
    """Scans the file to detect sheets. If single sheet, also detects columns."""
    worker = get_worker()
    fpath = state.get("file_path")
    
    if not fpath or not os.path.exists(fpath):
        return "Error: File path is missing or invalid."

    try:
        # Load data to get sheet names
        dfs = worker.load_data(fpath)
        sheet_names = list(dfs.keys())

        if len(sheet_names) == 1:
            # SINGLE SHEET: Scan columns immediately (preserves current behavior)
            sheet_name = sheet_names[0]
            df = dfs[sheet_name]
            return await _analyse_sheet_internal(worker, df, sheet_name, fpath)
        
        else:
            # MULTIPLE SHEETS: Show previews, ask user to pick
            preview_text = f"File scanned. Found **{len(sheet_names)} sheets**:\n\n"
            for name in sheet_names:
                df = dfs[name]
                row_count = len(df)
                col_preview = list(df.columns[:5])
                preview_text += f"- **{name}**: {row_count} rows, columns include: {col_preview}\n"
            
            preview_text += (
                "\nPROTOCOL: Multiple sheets detected. Present ALL sheet names clearly to the user. "
                "Ask the user: 'Which sheet contains the ticket data you want me to analyze?' "
                "Do NOT scan columns or trigger analysis yet — wait for the user to choose a sheet. "
                "When the user responds, use the `analyse_sheet` tool with the EXACT original sheet name."
            )
            return preview_text

    except Exception as e:
        return f"Error scanning file: {str(e)}"


async def tool_analyse_sheet(state: AgentState, sheet_name: str) -> str:
    """Analyses a specific sheet chosen by the user for column detection."""
    worker = get_worker()
    fpath = state.get("file_path")
    
    if not fpath or not os.path.exists(fpath):
        return "Error: File path is missing or invalid."
    
    try:
        dfs = worker.load_data(fpath)
        
        if sheet_name not in dfs:
            available = list(dfs.keys())
            return (
                f"Error: Sheet '{sheet_name}' not found. "
                f"Available sheets are: {available}. "
                "PROTOCOL: Tell the user the exact available sheet names and ask them to clarify."
            )
        
        df = dfs[sheet_name]
        return await _analyse_sheet_internal(worker, df, sheet_name, fpath)
    
    except Exception as e:
        return f"Error analysing sheet: {str(e)}"

async def tool_run_analysis(state: AgentState, sheet_name: str, id_col: str, summary_col: str, resolution_col: str = None) -> Dict:
    """Runs the full clustering analysis logic."""
    worker = get_worker()
    fpath = state.get("file_path")
    logs = []
    
    try:
        # 1. Load Data
        dfs = worker.load_data(fpath)
        if sheet_name not in dfs:
            if len(dfs) == 1:
                sheet_name = list(dfs.keys())[0]
            else:
                return {"status": "error", "output_text": f"Error: Sheet '{sheet_name}' not found. Available sheets: {list(dfs.keys())}", "logs": logs}
        df = dfs[sheet_name]
        
        # 2. Reconstruct Schema Object
        schema = SheetSchema(
            sheet_name=sheet_name, 
            id_column=id_col, 
            summary_column=summary_col, 
            resolution_column=resolution_col
        )
        
        # 3. Run Worker
        logs.append("Starting analysis pipeline...")
        results = await worker.run_full_analysis(df, schema, logs=logs)
        
        # 4. Setup SQLite (deterministic path using session_id)
        session_id = state.get("_session_id", "unknown")
        db_path = os.path.join(SQLITE_DIR, f"{session_id}.db")
        engine = create_engine(f"sqlite:///{db_path}")
        
        # 5. Save to SQL
        if results.get("individual_tickets") is not None:
            clean_df = results["individual_tickets"].copy()
            clean_df = _clean_df_columns(clean_df)
            clean_df.to_sql("ticket_data", engine, index=False, if_exists="replace")
            
        if results.get("cluster_report") is not None:
            results["cluster_report"].to_sql("cluster_summaries", engine, index=False, if_exists="replace")

        return {
            "status": "success",
            "df_individual": results["individual_tickets"],
            "df_report": results["cluster_report"],
            "executive_summary": results["executive_summary"],
            "db_engine": engine,
            "db_file_path": db_path,
            "logs": logs,
            "output_text": f"Analysis Complete.\n\nHere is your Executive Summary:\n{results['executive_summary']}\n\n**Detailed Use Case Report:**\nI have generated the Use Case Report table below."
        }
    except Exception as e:
        logger.error(f"Analysis Tool Failed: {e}")
        return {"status": "error", "output_text": f"Analysis failed: {str(e)}", "logs": logs}


# --- Chart Generation Helpers ---

# Professional color palette for charts (used by generate_default_chart only)
CHART_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# Column reference keys that the LLM uses in chart_config traces
# The tool replaces these with actual data arrays from the SQL result
_COLUMN_MAPPINGS = {
    "x_col": "x",
    "y_col": "y",
    "labels_col": "labels",
    "values_col": "values",
    "text_col": "text",
    "z_col": "z",
}

def _to_native(value):
    """Convert numpy types to native Python for JSON serialization."""
    if hasattr(value, 'item'):
        return value.item()
    return value


def build_chart_spec(engine, query: str, chart_config: dict):
    """
    Executes a SQL query and fills the LLM-generated Plotly chart config with actual data.
    
    The LLM provides the complete Plotly spec (data traces + layout) in chart_config.
    In data traces, it uses column reference keys (e.g., 'x_col': 'column_name') 
    instead of actual data arrays. This function replaces those references with 
    real data from the SQL query result.
    
    Args:
        engine: SQLAlchemy engine
        query: SQL query to get chart data
        chart_config: Complete Plotly spec with column references from the LLM
                      Expected format: {"data": [...traces...], "layout": {...}}
    
    Returns:
        (chart_spec_dict, text_summary_for_llm)
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        return None, f"Chart generation failed — SQL error: {e}"

    if df.empty:
        return None, "Chart generation failed — the query returned no data."

    str_columns = {str(c):c for c in df.columns}
    available_cols = list(str_columns.keys())
    
    # Deep copy to avoid mutating the original tool call args
    # (the original dict is stored in ToolMessage and pickled to Postgres)
    chart_config = copy.deepcopy(chart_config)
    
    # Substitute column references with actual data in each trace
    for trace in chart_config.get("data", []):
        for col_key, plotly_key in _COLUMN_MAPPINGS.items():
            if col_key in trace:
                col_name_str = str(trace.pop(col_key))
                if col_name_str in str_columns:
                    actual_col_name = str_columns[col_name_str]
                    values = df[actual_col_name].tolist()
                    trace[plotly_key] = [_to_native(v) for v in values]
                else:
                    return None, (
                        f"Chart generation failed — column '{col_name_str}' not found. "
                        f"Available columns: {available_cols}"
                    )

    num_rows = len(df)
    title = chart_config.get("layout", {}).get("title", {})
    title_text = title.get("text", "Chart") if isinstance(title, dict) else str(title)
    
    chart_spec = {"plotly": chart_config, "title": title_text}
    summary = (
        f"SUCCESS: Chart '{title_text}' generated with {num_rows} data points. "
        f"The chart has been sent to the user's screen. "
        f"Do NOT describe the chart data in detail — the user can already see it."
    )
    return chart_spec, summary


def generate_default_chart(cluster_report_df):
    """
    Generates 1 default chart from the cluster_report DataFrame.
    Used by background.py for the auto-chart after analysis completion.
    
    Returns:
        list with a single chart spec dict (wrapped in list for consistency with API response).
    """
    if cluster_report_df is None or cluster_report_df.empty:
        return []

    # Top Use Cases by Ticket Volume (horizontal bar)
    top = cluster_report_df.nlargest(10, "tickets_per_month")
    # Reverse for horizontal bar (largest on top)
    top = top.iloc[::-1]

    num_points = len(top)
    colors = (CHART_COLORS * ((num_points // len(CHART_COLORS)) + 1))[:num_points]

    chart = {
        "title": "Top Use Cases by Ticket Volume",
        "chart_type": "horizontal_bar",
        "plotly": {
            "data": [{
                "y": top["usecase_name"].tolist(),
                "x": [float(v) for v in top["tickets_per_month"].tolist()],
                "type": "bar",
                "orientation": "h",
                "marker": {"color": colors}
            }],
            "layout": {
                "title": {"text": "Top Use Cases by Ticket Volume"},
                "xaxis": {"title": "Tickets Per Month"},
                "yaxis": {"automargin": True},
                "margin": {"l": 250}
            }
        }
    }

    return [chart]