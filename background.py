"""
Background Task Manager for long-running analysis jobs.

Uses asyncio.create_task() to run analysis in the background while
returning an immediate HTTP response to the user. This works because
gunicorn is run with -w 1 (single worker), so the asyncio event loop
is shared across all requests in the same process.

Task status is tracked in-memory (dict keyed by session_id).
"""

import os
import asyncio
from typing import Optional, Dict, Any
from config import logger

# In-memory task status store: session_id -> {status, error}
_task_status: Dict[str, Dict[str, Any]] = {}

# Track asyncio Task objects to prevent garbage collection
_task_refs: Dict[str, asyncio.Task] = {}

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


def get_task_status(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Returns the current status of a background analysis task.
    
    Returns:
        dict with {"status": "running"|"completed"|"failed", "error": str|None}
        or None if no task has been started for this session.
    """
    return _task_status.get(session_id)


def is_analysis_running(session_id: str) -> bool:
    """Check if a background analysis is currently running for this session."""
    info = _task_status.get(session_id)
    return info is not None and info["status"] == "running"


def cancel_analysis_task(session_id: str) -> bool:
    """
    Cancels a running background analysis task for the given session.
    Called when user deletes a session to stop wasting tokens and compute.

    Returns:
        True if a running task was found and cancelled, False otherwise.
    """
    task = _task_refs.get(session_id)
    if task and not task.done():
        task.cancel()
        logger.info(f"Cancelled background analysis task for session {session_id}")

    # Clean up tracking state regardless (handles edge cases where task just finished)
    _task_refs.pop(session_id, None)
    _task_status.pop(session_id, None)
    return task is not None


def start_analysis_task(
    session_id: str,
    state: dict,
    sheet_name: str,
    id_col: str,
    summary_col: str,
    resolution_col: Optional[str],
    session_manager: Any,
    user_email_hash: Optional[str] = None,
    duration_months: int = 1,
    ticket_type_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Launches the analysis as a background asyncio task.

    Args:
        session_id: The session to run analysis for.
        state: Current agent state dictionary.
        sheet_name, id_col, summary_col, resolution_col: Analysis parameters.
        session_manager: SessionManager instance for saving results.
        user_email_hash: Hashed user email for session ownership.

    Returns:
        True if task was started, False if one is already running.
    """
    # Prevent duplicate runs for the same session
    if is_analysis_running(session_id):
        logger.warning(f"Analysis already running for session {session_id}")
        return False

    # Mark as running immediately
    _task_status[session_id] = {"status": "running", "error": None}

    # Create the background task
    task = asyncio.create_task(
        _run_analysis_wrapper(
            session_id, state, sheet_name, id_col,
            summary_col, resolution_col, session_manager,
            user_email_hash, duration_months, ticket_type_config
        )
    )

    # Store reference to prevent garbage collection
    _task_refs[session_id] = task

    # Clean up reference when task completes
    task.add_done_callback(lambda t: _task_refs.pop(session_id, None))

    logger.info(f"Background analysis started for session {session_id}")
    return True


async def _run_analysis_wrapper(
    session_id: str,
    state: dict,
    sheet_name: str,
    id_col: str,
    summary_col: str,
    resolution_col: Optional[str],
    session_manager: Any,
    user_email_hash: Optional[str] = None,
    duration_months: int = 1,
    ticket_type_config: Optional[Dict[str, Any]] = None
):
    """
    Wrapper coroutine that runs the full analysis pipeline in the background.
    On completion (success or failure), saves results to Postgres.
    """
    import re
    import pandas as pd
    from sqlalchemy import create_engine
    from langchain_core.messages import AIMessage
    from tools import get_worker, generate_default_chart
    from models import SheetSchema
    from config import SQLITE_DIR

    logs = []

    try:
        worker = get_worker()
        fpath = state.get("file_path")

        # 1. Load Data (offload file I/O to thread)
        dfs = await asyncio.to_thread(worker.load_data, fpath)
        if sheet_name not in dfs:
            if len(dfs) == 1:
                sheet_name_actual = list(dfs.keys())[0]
            else:
                raise ValueError(f"Sheet '{sheet_name}' not found. Available: {list(dfs.keys())}")
        else:
            sheet_name_actual = sheet_name

        df = dfs[sheet_name_actual]

        # 2. Build Schema
        schema = SheetSchema(
            sheet_name=sheet_name_actual,
            id_column=id_col,
            summary_column=summary_col,
            resolution_column=resolution_col
        )

        # 3. Run Full Analysis
        logs.append("Starting analysis pipeline...")
        results = await worker.run_full_analysis(df, schema, logs=logs, duration_months=duration_months, ticket_type_config=ticket_type_config)

        # 4. Setup SQLite (deterministic path using session_id)
        db_path = os.path.join(SQLITE_DIR, f"{session_id}.db")
        engine = create_engine(f"sqlite:///{db_path}")

        # 5. Save to SQL (offload SQLite writes to thread)
        if results.get("individual_tickets") is not None:
            clean_df = results["individual_tickets"].copy()
            # clean_df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(c)) for c in clean_df.columns]
            clean_df = _clean_df_columns(clean_df)
            await asyncio.to_thread(clean_df.to_sql, "ticket_data", engine, index=False, if_exists="replace")

        if results.get("cluster_report") is not None:
            await asyncio.to_thread(results["cluster_report"].to_sql, "cluster_summaries", engine, index=False, if_exists="replace")

        # 6. Build the completed state
        output_text = (
            f"Analysis Complete.\n\nHere is your Executive Summary:\n"
            f"{results['executive_summary']}\n\n"
            f"**Detailed Use Case Report:**\n"
            f"I have generated the Use Case Report table below."
        )

        # 7. Generate auto-chart from cluster report
        default_charts = generate_default_chart(results["cluster_report"])

        # Build the updated state — start from the original state
        latest_state = await asyncio.to_thread(session_manager.load_session, session_id)

        # GUARD: If the session was deleted while analysis was running, abort save.
        # Without this, the upsert in save_session would re-create the deleted row.
        if latest_state is None:
            logger.warning(f"Session {session_id} was deleted during analysis — skipping save.")
            _task_status[session_id] = {"status": "failed", "error": "Session was deleted."}
            return

        result_state = latest_state
        
        result_state.pop('_session_id', None)  # strip runtime-only fields
        result_state.pop('_user_email_hash', None)
        result_state["df_individual"] = results["individual_tickets"]
        result_state["df_report"] = results["cluster_report"]
        result_state["executive_summary"] = results["executive_summary"]
        result_state["db_engine"] = engine
        result_state["db_file_path"] = db_path
        result_state["analysis_complete"] = True
        result_state["analysis_in_progress"] = False
        result_state["modified_df"] = results["cluster_report"]
        result_state["chart_data"] = default_charts
        result_state["logs"] = logs

        # NOTE: Do NOT overwrite session_title here.
        # The title was set during upload (from filename). If the user renamed it,
        # latest_state already has the correct title. We preserve it as-is.

        # 8. Add an AI message with the results so it shows in history
        from langchain_core.messages import HumanMessage, ToolMessage

        ai_result_msg = AIMessage(
            content=output_text,
            additional_kwargs={"charts": default_charts if default_charts else [],
                               "dataframe":results["cluster_report"]
            }
        )
        result_state["messages"] = list(result_state.get("messages", [])) + [ai_result_msg]

        # 9. Save to Postgres (offload to thread)
        # Pass title=None so SQL COALESCE preserves the existing DB title
        await asyncio.to_thread(
            session_manager.save_session, session_id, result_state,
            None, user_email_hash
        )

        # 10. Mark task as completed
        _task_status[session_id] = {"status": "completed", "error": None}
        logger.info(f"Background analysis COMPLETED for session {session_id}")

    except asyncio.CancelledError:
        # Task was cancelled (user deleted the session) — clean exit, no save needed
        logger.info(f"Background analysis CANCELLED for session {session_id}")
        _task_status.pop(session_id, None)
        return

    except Exception as e:
        # Strip SQL query details from error message (security: don't log raw SQL/params)
        error_msg = str(e).split('\n[SQL:')[0].strip()
        logger.error(f"Background analysis FAILED for session {session_id}: {error_msg}", exc_info=True)
        _task_status[session_id] = {"status": "failed", "error": error_msg}

        # Save the failure state so user sees the error
        try:
            # 1. Brief pause to guarantee a2a_main.py has committed the ToolMessage to Postgres
            await asyncio.sleep(1.0)
            
            # 2. Fetch the absolute latest state, do NOT use the stale memory state
            latest_state = await asyncio.to_thread(session_manager.load_session, session_id)

            # GUARD: If the session was deleted, don't re-create it with error state
            if latest_state is None:
                logger.warning(f"Session {session_id} was deleted during analysis — skipping error state save.")
                return

            fail_state = latest_state

            fail_state.pop('_session_id', None)  # strip runtime-only fields
            fail_state.pop('_user_email_hash', None)
            fail_state["analysis_in_progress"] = False
            fail_state["logs"] = logs + [f"❌ Analysis failed: {error_msg}"]

            fail_msg = AIMessage(content=f"I'm sorry, the analysis failed with an error: {error_msg}. Please try uploading the file again.")
            fail_state["messages"] = list(fail_state.get("messages", [])) + [fail_msg]

            await asyncio.to_thread(
                session_manager.save_session, session_id, fail_state,
                None, user_email_hash
            )
        except Exception as save_err:
            logger.error(f"Failed to save error state for session {session_id}: {type(save_err).__name__}")