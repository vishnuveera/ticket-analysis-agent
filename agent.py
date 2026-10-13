import re
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from sqlalchemy import inspect as sql_inspect

# Local Imports
from config import (
    AZURE_DEPLOYMENT, AZURE_MINI_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT,
    AZURE_API_VERSION, AZURE_ENDPOINT, logger, get_azure_api_key
)
from models import AgentState
import tools
from tools import build_chart_spec
import background

# This will be set by main.py at startup so the background task can save results
_session_manager = None

def set_session_manager(sm):
    """Called by main.py to inject the session manager reference."""
    global _session_manager
    _session_manager = sm


# --- Token-Aware Sliding Window ---
# Prevents context overflow on long sessions by trimming older messages
# before sending to LLM, while keeping full history intact in state.

def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: ~1 token per 4 characters."""
    return len(str(text)) // 4

def _estimate_message_tokens(msg) -> int:
    """Estimates token count for a single LangChain message."""
    tokens = _estimate_tokens(msg.content or "")
    # Account for tool_calls JSON payload in AIMessages
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            tokens += _estimate_tokens(str(tc.get('args', '')))
    return tokens

def trim_messages_for_llm(messages: list, max_tokens: int = 100000) -> list:
    """
    Trims old messages to fit within the LLM's context window.
    
    Strategy:
    - Always keep the FIRST 4 messages (initial scan/analysis setup context)
    - Always keep the LAST 10 messages (recent conversation)
    - If estimated tokens exceed max_tokens, progressively drop older 
      middle messages, replacing ToolMessages with compact summaries
    - Never mutates the original list
    
    Returns a new list of messages safe for LLM invocation.
    """
    if not messages:
        return messages
    
    KEEP_FIRST = 4
    KEEP_LAST = 10
    
    # If the history is small enough, skip trimming entirely
    if len(messages) <= KEEP_FIRST + KEEP_LAST:
        return list(messages)
    
    # Estimate total tokens
    total_tokens = sum(_estimate_message_tokens(m) for m in messages)
    
    if total_tokens <= max_tokens:
        return list(messages)
    
    # --- Trimming needed ---
    logger.info(
        f"[SLIDING WINDOW] History has ~{total_tokens} tokens (limit: {max_tokens}). "
        f"Trimming {len(messages)} messages."
    )
    
    head = list(messages[:KEEP_FIRST])
    tail = list(messages[-KEEP_LAST:])
    middle = messages[KEEP_FIRST:-KEEP_LAST]
    
    # Calculate budget: how many tokens we can afford for middle messages
    head_tokens = sum(_estimate_message_tokens(m) for m in head)
    tail_tokens = sum(_estimate_message_tokens(m) for m in tail)
    budget = max_tokens - head_tokens - tail_tokens
    
    # Greedily keep middle messages from newest to oldest (newest = most relevant)
    kept_middle = []
    used_tokens = 0
    
    for msg in reversed(middle):
        msg_tokens = _estimate_message_tokens(msg)
        
        # ToolMessages with large content are the biggest targets
        if isinstance(msg, ToolMessage) and msg_tokens > 500:
            # Replace with a compact summary to preserve tool_call_id linkage
            row_match = re.search(r'Result contains (\d+) rows', msg.content or '')
            row_count = row_match.group(1) if row_match else "N"
            compact = ToolMessage(
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, 'name', ''),
                content=f"[Previous query returned {row_count} rows — context trimmed for efficiency]"
            )
            compact_tokens = _estimate_message_tokens(compact)
            if used_tokens + compact_tokens <= budget:
                kept_middle.append(compact)
                used_tokens += compact_tokens
            # else: drop entirely
        elif used_tokens + msg_tokens <= budget:
            kept_middle.append(msg)
            used_tokens += msg_tokens
        # else: drop this message
    
    # Reverse back to chronological order
    kept_middle.reverse()
    
    trimmed = head + kept_middle + tail
    logger.info(
        f"[SLIDING WINDOW] Trimmed to {len(trimmed)} messages (~{head_tokens + used_tokens + tail_tokens} tokens)"
    )
    return trimmed


async def node_tool_executor(state: AgentState) -> AgentState:
    """Executes the tool call requested by the Assistant."""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    
    results = []
    updated_state = {
        "modified_df":None,
        "chart_data":None,
    }

    for tool_call in tool_calls:
        t_name = tool_call["name"]
        t_args = tool_call["args"]
        t_id = tool_call["id"]
        
        output_str = ""
        
        try:
            if t_name == "scan_file":
                output_str = await tools.tool_scan_file(state)

            elif t_name == "analyse_sheet":
                output_str = await tools.tool_analyse_sheet(state, t_args.get("sheet_name", ""))
                
            elif t_name == "trigger_analysis":
                session_id = state.get("_session_id")  
                user_email_hash = state.get("_user_email_hash")  
                
                started = background.start_analysis_task(
                    session_id=session_id,
                    state=state,
                    sheet_name=t_args.get("sheet_name"),
                    id_col=t_args.get("id_col"),
                    summary_col=t_args.get("summary_col"),
                    resolution_col=t_args.get("resolution_col"),
                    duration_months=t_args.get("duration_months", 1),
                    session_manager=_session_manager,
                    user_email_hash=user_email_hash,
                    ticket_type_config=t_args.get("ticket_type_config")
                )
                
                if started:
                    updated_state["analysis_in_progress"] = True
                    output_str = "BACKGROUND_TASK_STARTED" # Agent won't see this due to short-circuit
                else:
                    # Known failure (already running). Agent WILL read this.
                    output_str = (
                        "SYSTEM INSTRUCTION: The analysis is already running in the background. "
                        "Politely tell the user to wait for it to complete."
                    )

            elif t_name == "execute_sql":
                engine = state.get("db_engine")
                if not engine:
                    output_str = "Error: Database is not active. Please re-run analysis."
                else:
                    container = {} 
                    sql_tool = tools.create_sql_tool(engine, container)
                    output_str = sql_tool.invoke(t_args["query"])
                
                    if container.get("modified_df") is not None:
                        updated_state["modified_df"] = container["modified_df"]

            elif t_name == "generate_chart":
                engine = state.get("db_engine")
                if not engine:
                    output_str = "Error: Database is not active. Please re-run analysis."
                else:
                    chart_spec, summary_text = build_chart_spec(
                        engine=engine,
                        query=t_args["query"],
                        chart_config=t_args["chart_config"]
                    )
                    if chart_spec:
                        updated_state["chart_data"] = [chart_spec]
                        output_str = summary_text

        except Exception as e:
            # 1. Log the actual technical error to your Azure App Service logs
            logger.error(f"[TOOL EXECUTION ERROR] Failed in tool '{t_name}': {str(e)}", exc_info=True)
            
            # 2. Give the agent a generalized instruction
            output_str = (
                "SYSTEM INSTRUCTION: An internal system error occurred while running this tool. "
                "Do NOT expose technical details or code. Politely apologize to the user and suggest they try again later."
            )

        msg_kwargs = {}
        if updated_state.get("modified_df") is not None:
             msg_kwargs["dataframe"] = updated_state["modified_df"]
        if updated_state.get("chart_data") is not None:
             msg_kwargs["chart"] = updated_state["chart_data"]
             
        results.append(ToolMessage(tool_call_id=t_id, name=t_name, content=str(output_str), additional_kwargs=msg_kwargs))

    return {
        "messages": results,
        **updated_state
    }

# Node Assistant

async def node_assistant(state: AgentState) -> dict:
    """The brain: Decides whether to chat or call a tool."""

    if state.get("messages"):
        last_msg = state["messages"][-1]
        
        # Check if the last message was the trigger_analysis tool AND it succeeded
        if getattr(last_msg, "name", None) == "trigger_analysis":
            if state.get("analysis_in_progress") == True:
                static_text = "The analysis has started and is running in the background. It will take a few minutes to complete. You can check back shortly."
                return {"messages": [AIMessage(content=static_text)]}
    
    # 1. Define Tools Schema
    tools = [
        {
            "name": "scan_file",
            "description": "Scans the uploaded file to detect sheets. If single sheet, also detects columns. If multiple sheets, lists them for user selection. Use this immediately when a file is present but analysis hasn't run.",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "analyse_sheet",
            "description": "Analyses a specific sheet for column detection. Use this ONLY after scan_file reports multiple sheets and the user has selected one. Pass the EXACT original sheet name as reported by scan_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {"type": "string", "description": "The EXACT original sheet name as reported by scan_file"}
                },
                "required": ["sheet_name"]
            }
        },
        {
            "name": "trigger_analysis",
            "description": "Runs the analysis. Use ONLY after user confirms columns and ticket type configuration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sheet_name": {"type": "string", "description": "The exact name of the sheet to analyze"},
                    "id_col": {"type": "string"},
                    "summary_col": {"type": "string"},
                    "resolution_col": {"type": "string"},
                    "duration_months": {"type": "integer", "description": "The number of months of data. Default is 1."},
                    "ticket_type_config": {
                        "type": "object",
                        "description": "Configuration for unknown ticket types detected during scan. Each key is a prefix (e.g., 'CRQ', 'TASK', 'NUMERIC'). Each value is an object with 'type_name' (string) and 'effort_hours' (number). Only include this if unknown ticket types were detected during scan and the user has confirmed them. Example: {'CRQ': {'type_name': 'Change Request', 'effort_hours': 3}, 'NUMERIC': {'type_name': 'General Ticket', 'effort_hours': 2}}"
                    }
                },
                "required": ["id_col", "summary_col"]
            }
        }
    ]
    
    if state.get("analysis_complete") and state.get("db_engine"):
        
        inspector = sql_inspect(state["db_engine"])
        all_schemas = []
        for table in inspector.get_table_names():
             cols = inspector.get_columns(table)
             cols_str = ", ".join([f"{c['name']} ({c['type']})" for c in cols])
             all_schemas.append(f"Table '{table}': {cols_str}")
        
        schema_text = "\n".join(all_schemas)
        
        tools.append({
            "name": "execute_sql",
            "description": f"Query the ticket data. You have two tables linked by 'usecase_id'.\nAvailable Schemas:\n{schema_text}",
            "parameters": {
                "type": "object", 
                "properties": {"query": {"type": "string"}}, 
                "required": ["query"]
            }
        })
        tools.append({
            "name": "generate_chart",
            "description": (
                f"Generates a chart visualization from the ticket data. Use when user asks for charts, graphs, or visual breakdowns.\n"
                f"Available Schemas:\n{schema_text}\n\n"
                f"CONVENTION: In 'chart_config.data' traces, use column references instead of data arrays:\n"
                f"  - 'x_col': 'column_name'  → replaced with actual x-axis data\n"
                f"  - 'y_col': 'column_name'  → replaced with actual y-axis data\n"
                f"  - 'labels_col': 'column_name' → for pie/donut charts\n"
                f"  - 'values_col': 'column_name' → for pie/donut charts\n"
                f"  - 'text_col': 'column_name' → for text annotations\n"
                f"All other Plotly trace properties (type, marker, orientation, mode, etc.) are passed through as-is."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to get the data for the chart"},
                    "chart_config": {
                        "type": "object",
                        "description": "Complete Plotly chart specification with column references. Must contain 'data' (array of traces) and 'layout' (object).",
                        "properties": {
                            "data": {
                                "type": "array", 
                                "description": "Array of Plotly trace objects with column references (x_col, y_col, etc.)",
                                "items":{"type":"object"}
                                     },
                            "layout": {"type": "object", "description": "Plotly layout object (title, axes, margins, etc.)"}
                        },
                        "required": ["data", "layout"]
                    }
                },
                "required": ["query", "chart_config"]
            }
        })

    # 2. System Prompt
    system_msg = """
    You are an expert IT Ticket Analysis Agent.
    
    **Current Status:**
    - File Uploaded: {has_file}
    - Analysis Complete: {is_complete}

    You must analyze the user's input AND the conversation history to choose the right tool. Follow this logic in order:
    1.  **CASE A: SECURITY BLOCK / OFF-TOPIC (Highest Priority)**
          * *Scenario:* The user asks about unrelated topics (e.g., recipes, jokes, general knowledge).
          * *Action:* **DO NOT use any tool.** Refuse the request politely.
          * *Reasoning:* You must remain within your domain and maintain security.

    **INSTRUCTIONS & RULES:**
    1.  **SECURITY & GUARDRAILS (Strict Enforcement):**
     * **Domain Restriction:** Stay within the IT ticket analysis scope. If a user asks about unrelated topics (such as recipes, travel, sports, or general knowledge), politely explain that you are designed for IT ticket analysis and cannot help with unrelated requests.*
    
    **PROTOCOL:**
    1. If `Analysis Complete` is False:
       - You MUST call `scan_file` first.
       - **MULTI-SHEET HANDLING:** If scan_file reports multiple sheets:
         a. Present ALL sheet names clearly to the user with their previews.
         b. Ask: "Which sheet contains the ticket data you want me to analyze?"
         c. When the user responds, YOU must resolve their response to the EXACT original sheet name 
            from the scan results. The user may say things like "the second one", "data sheet", 
            "Sheet2", "sheet 2", abbreviations, or slightly different casing — use your intelligence 
            to match their intent to the correct exact sheet name as listed in the scan output.
         d. Call `analyse_sheet` with that EXACT original sheet name.
         e. Do NOT scan columns or trigger analysis until the user picks a sheet.
       - **SINGLE SHEET:** scan_file will directly show columns — proceed to confirm below.
       - Present columns to user. Ask "Is this correct?" AND ask "How many months of data does this file contain?".
       - **COLUMN NAME RESOLUTION:** When the user confirms or corrects columns, resolve their input 
         to the EXACT original column names from the scan results. Users may paraphrase, abbreviate, 
         use different casing, or skip special characters — match intelligently to the correct column name. 
         Always use the exact original column name when calling `trigger_analysis`.
       - **UNKNOWN TICKET TYPES:** If the scan detected unknown ticket type prefixes:
         a. Present the detected unknown prefixes to the user with counts and the LLM's suggested classification.
         b. Ask the user to confirm or correct the type name for each unknown prefix.
         c. Ask the user for the estimated effort hours per ticket for each unknown type.
         d. When calling `trigger_analysis`, include the user-confirmed types in the `ticket_type_config` parameter.
            Format: {{{{"PREFIX": {{{{"type_name": "Confirmed Name", "effort_hours": N}}}}}}}}
            Example: {{{{"CRQ": {{{{"type_name": "Change Request", "effort_hours": 3}}}}, "NUMERIC": {{{{"type_name": "General Ticket", "effort_hours": 2}}}}}}}}
       - If NO unknown types were detected, proceed normally (no `ticket_type_config` needed).
       - If user says YES and provides the months: Call `trigger_analysis` with the columns, `duration_months`, and `ticket_type_config` (if applicable).
       - If user says NO/CORRECTS: Call `trigger_analysis` with NEW columns and the specified `duration_months`.
        
    2. If `Analysis Complete` is True:
       - Answer user questions using `execute_sql`.
       - **IMPORTANT:** You have two tables: `ticket_data` (Individual tickets) and `cluster_summaries` (High-level report).
       - To filter tickets by a specific topic (e.g., "Outlook"), you often need to JOIN these tables on `usecase_id` or query `cluster_summaries` first to get the ID.
       - Never answer from memory. Always query the database.
        
    3. **CHARTS & VISUALIZATION:**
       - When user asks for charts, graphs, visualizations, or visual breakdowns, use `generate_chart`.
       - You generate the COMPLETE Plotly chart specification dynamically — you have full control over chart type, styling, colors, and layout.
       - In `chart_config.data` traces, use column references (e.g., `x_col`, `y_col`, `labels_col`, `values_col`) — the tool replaces them with actual data from the SQL result.
       - You can create ANY Plotly chart type: bar, pie, scatter, line, funnel, histogram, treemap, heatmap, waterfall, etc.
       - If user asks to change chart type, generate a new `chart_config` with the desired type.
       - Use professional colors in your chart specs.
       - **CRITICAL RULE FOR DATA + CHART REQUESTS:**
            - 'generate_chart' ONLY produces a visual chart — it does NOT return or display the underlying data table to the user.
            - If the user asks for BOTH data AND a chart/visualization (e.g., "show me top 10 usecases and a pie chart"), you MUST call BOTH 'execute_sql' (to display the data table) AND 'generate_chart' (to display the chart) as parallel tool calls in the SAME turn.
            - If the user asks for ONLY a chart/visualization (e.g., "show me a pie chart"), use ONLY 'generate_chart'.
            - If the user asks for ONLY data (e.g., "show me the top 10"), use ONLY 'execute_sql'.
    """
    
    has_file = "Yes" if state.get("file_path") else "No"
    is_complete = "Yes" if state.get("analysis_complete") else "No"
    
    formatted_sys = system_msg.format(has_file=has_file, is_complete=is_complete)

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_sys),
        ("placeholder", "{messages}")
    ])

    llm = AzureChatOpenAI(
        deployment_name=AZURE_DEPLOYMENT, 
        openai_api_version=AZURE_API_VERSION, 
        azure_endpoint=AZURE_ENDPOINT,
        api_key=get_azure_api_key(),
        temperature=0
    ).bind_tools(tools)

    chain = prompt | llm
    
    # Apply sliding window: trim old messages for LLM context only
    # Full history in state["messages"] remains untouched for UI/persistence
    trimmed_messages = trim_messages_for_llm(state["messages"])
    response = await chain.ainvoke({"messages": trimmed_messages})
    return {"messages": [response]}

def route_agent(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if not last_msg.tool_calls:
        return END
    return "tool_executor"

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("assistant", node_assistant)
    workflow.add_node("tool_executor", node_tool_executor)

    workflow.add_edge(START, "assistant")
    workflow.add_conditional_edges("assistant", route_agent, ["tool_executor", END])
    workflow.add_edge("tool_executor", "assistant")
    
    return workflow.compile()