import os
import io
import glob
import time
import asyncio
import shutil
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from starlette.responses import JSONResponse

# A2A imports — wrapper module provides the A2A Starlette app + executor
from a2a_wrapper import a2a_app, ticket_agent_card, TicketA2AExecutor

from config import (
    API_KEY_NAME, get_auth_secret, UPLOAD_DIR, SQLITE_DIR, DATA_RETENTION_DAYS,
    get_pg_connection_string, DB_SCHEMA, DB_TABLE, logger
)
from models import ChatRequest, ApiResponse, ContentPayload, TitleUpdateRequest, AgentState
from session import SessionManager, rehydrate_sqlite_from_state, hash_email
from agent import build_graph, set_session_manager
import background

# --- Globals ---
session_manager = SessionManager(get_pg_connection_string(), DB_SCHEMA, DB_TABLE)

# Inject session_manager into agent module so background tasks can save results
set_session_manager(session_manager)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
async def get_api_key(key: str = Depends(api_key_header)):
    if key == get_auth_secret(): return key
    raise HTTPException(403, "Invalid API Key")

async def get_user_email_hash(x_user_email: str = Header(None)):
    """Extracts and hashes the user email from the X-User-Email header."""
    if not x_user_email:
        raise HTTPException(401, "Missing X-User-Email header")
    return hash_email(x_user_email)

# --- API App Definition ---

app = FastAPI(title="Ticket Analysis Agent API")

# --- A2A INTEGRATION BLOCK ---

# 1. Inject the session manager into the A2A Executor
TicketA2AExecutor.set_session_manager(session_manager)

# 2. Mount the A2A App
app.mount("/a2a", a2a_app.build())

# 3. Expose the Agent Card for Discovery (standard A2A well-known endpoint)
@app.get("/.well-known/ai-agent.json")
async def get_discovery():
    """Endpoint for A2A agent discovery."""
    return JSONResponse(ticket_agent_card.model_dump(mode="json", by_alias=True))

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Daily Disk Cleanup ---

async def _daily_disk_cleanup():
    """Runs once per day, deletes files older than DATA_RETENTION_DAYS."""
    while True:
        await asyncio.sleep(86400)  # Wait 24 hours between runs
        try:
            cutoff = time.time() - (DATA_RETENTION_DAYS * 86400)
            deleted_count = 0

            for directory in [SQLITE_DIR, UPLOAD_DIR]:
                for filepath in glob.glob(os.path.join(directory, "*")):
                    if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        deleted_count += 1

            logger.info(f"Disk cleanup complete. Deleted {deleted_count} files older than {DATA_RETENTION_DAYS} days.")
        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Launches the daily cleanup task on server start."""
    asyncio.create_task(_daily_disk_cleanup())
    logger.info("Daily disk cleanup scheduler started.")

# --- REST Endpoints (Original FastAPI — kept for backward compatibility) ---

@app.post("/upload", dependencies=[Depends(get_api_key)])
async def upload_file(session_id: str, file: UploadFile = File(...), user_hash: str = Depends(get_user_email_hash)):
    try:
        fpath = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
        with open(fpath, "wb+") as f:
            shutil.copyfileobj(file.file, f)

        clean_title = file.filename.rsplit('.',1)[0].replace('_',' ').replace('-',' ').title()

        # Deduplicate session title (e.g., "Course" → "Course 1" if "Course" exists)
        existing_sessions = session_manager.list_sessions(user_hash)
        existing_titles = {s.get("title") for s in existing_sessions}
        final_title = clean_title
        suffix_counter = 1
        while final_title in existing_titles:
            final_title = f"{clean_title} {suffix_counter}"
            suffix_counter += 1

        # Initialize Default State
        initial_state = AgentState(
            messages=[],
            file_path=fpath,
            df_individual=None,
            df_report=None,
            executive_summary=None,
            db_engine=None,
            modified_df=None,
            chart_data=None,
            db_file_path=None,
            analysis_complete=False,
            analysis_in_progress=False,
            session_title=final_title,
            logs=[],
            _session_id=None,
            _user_email_hash=None
        )

        session_manager.save_session(session_id, initial_state, title=final_title, user_email_hash=user_hash)

        # Enforce rolling session limit: delete oldest sessions if user exceeds the cap
        session_manager.enforce_session_limit(user_hash)

        return {"status": "success", "message": "File uploaded and processed."}

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, "An internal server error occurred while processing your request. Please try again later.")

@app.post("/chat", dependencies=[Depends(get_api_key)])
async def chat(req: ChatRequest, user_hash: str = Depends(get_user_email_hash)):
    # 1. Load Session
    state = session_manager.load_session(req.session_id)
    if not state:
        raise HTTPException(404, "Session not found")

    # 1.5 Verify ownership
    if not session_manager.verify_ownership(req.session_id, user_hash):
        raise HTTPException(403, "Access denied")

    # 2. Hydrate SQLite (Re-connect to local DB file)
    state = rehydrate_sqlite_from_state(state)

    state['modified_df'] = None
    state['chart_data'] = None

    # 2.5 Inject runtime fields so the agent can pass them to background tasks
    state['_session_id'] = req.session_id
    state['_user_email_hash'] = user_hash

    # 3. Add User Message
    state['messages'].append(HumanMessage(content=req.message))

    # 4. Run Graph
    try:
        app_graph = build_graph()
        result = await app_graph.ainvoke(state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}", exc_info=True)
        raise HTTPException(500, "An internal server error occurred while processing your request. Please try again later.")

    # 5. Save State (strip runtime-only fields)
    result.pop('_session_id', None)
    result.pop('_user_email_hash', None)
    current_title = result.get("session_title")
    session_manager.save_session(req.session_id, result, title=current_title)

    # 6. Format Response
    last_msg = result['messages'][-1]

    data_payload = None
    if result.get('modified_df') is not None:
        df = result['modified_df']
        if isinstance(df, pd.DataFrame):
            df_clean = df.replace([np.inf,-np.inf], np.nan).fillna("")
            data_payload = df_clean.astype(str).to_dict(orient="records")
            result['modified_df'] = None

    # 6b. Extract Chart Data (same pattern as modified_df)
    chart_payload = None
    if result.get('chart_data') is not None:
        chart_payload = result['chart_data']
        if not isinstance(chart_payload, list):
            chart_payload = [chart_payload]
        result['chart_data'] = None

    # Save state once after clearing transient fields
    if data_payload is not None or chart_payload is not None:
        session_manager.save_session(req.session_id, result, title=current_title)

    return ApiResponse(
        session_id=req.session_id,
        status="success",
        content=ContentPayload(text=last_msg.content, data=data_payload, charts=chart_payload),
        execution_logs=result.get("logs", []),
        session_title=current_title
    )

@app.get("/history/{session_id}", dependencies=[Depends(get_api_key)])
async def get_history(session_id: str, user_hash: str = Depends(get_user_email_hash)):
    state = session_manager.load_session(session_id)
    if not state: raise HTTPException(404, "Session not found")

    # Verify ownership
    if not session_manager.verify_ownership(session_id, user_hash):
        raise HTTPException(403, "Access denied")

    # 1. Recover Context (File Name & Analysis Status)
    analysis_complete = state.get("analysis_complete", False)
    file_path = state.get("file_path")
    file_name = None
    if file_path:
        base_name = os.path.basename(file_path)
        parts = base_name.split('_', 1)
        file_name = parts[1] if len(parts) > 1 else base_name

    raw_messages = state.get("messages", [])
    formatted_messages = []

    # 2. Message Reconstruction Loop
    buffered_data = None
    buffered_chart = None

    for msg in raw_messages:
        role = "unknown"
        data_payload = None
        chart_payload = None
        content = ""

        if isinstance(msg, HumanMessage):
            # --- HIDE SYSTEM TRIGGERS ---
            if msg.content.startswith("SYSTEM_EVENT"):
                continue
            role = "user"
            content = msg.content

        elif isinstance(msg, ToolMessage):
            # --- CAPTURE DATAFRAMES ---
            if "dataframe" in msg.additional_kwargs:
                df = msg.additional_kwargs["dataframe"]
                if isinstance(df, pd.DataFrame):
                    df_clean = df.replace([np.inf,-np.inf], np.nan).fillna("")
                    buffered_data = df_clean.astype(str).to_dict(orient="records")
            # --- CAPTURE CHARTS (on-demand via generate_chart tool) ---
            if "chart" in msg.additional_kwargs:
                buffered_chart = msg.additional_kwargs["chart"]
            continue

        elif isinstance(msg, AIMessage):
            if getattr(msg,'tool_calls',None):
                continue

            role = "assistant"
            content = msg.content
            if buffered_data:
                data_payload = buffered_data
                buffered_data = None
            if buffered_chart:
                chart_payload = buffered_chart
                buffered_chart = None
            # --- CAPTURE AUTO-CHARTS (from background analysis completion) ---
            if "charts" in msg.additional_kwargs:
                chart_payload = msg.additional_kwargs["charts"]

            if "dataframe" in msg.additional_kwargs:
                df = msg.additional_kwargs["dataframe"]
                if isinstance(df,pd.DataFrame):
                    df_clean = df.replace([np.inf,-np.inf],np.nan).fillna("")
                    data_payload = df_clean.astype(str).to_dict(orient="records")

            if role == 'assistant' and not content and not data_payload and not chart_payload:
                continue

        if role in ["user", "assistant"]:
            formatted_messages.append({
                "role": role,
                "content": content,
                "data": data_payload,
                "charts": chart_payload,
                "logs": []
            })

    # Check if analysis is in progress (from saved state or live background task)
    analysis_in_progress = state.get("analysis_in_progress", False)
    if background.is_analysis_running(session_id):
        analysis_in_progress = True

    return {
        "session_id": session_id,
        "messages": formatted_messages,
        "logs": state.get("logs", []),
        "analysis_complete": analysis_complete,
        "analysis_in_progress": analysis_in_progress,
        "file_name": file_name
    }


@app.get("/download_workbook/{session_id}", dependencies=[Depends(get_api_key)])
async def download_result(session_id: str, user_hash: str = Depends(get_user_email_hash)):
    # Verify ownership
    if not session_manager.verify_ownership(session_id, user_hash):
        raise HTTPException(403, "Access denied")

    state = session_manager.load_session(session_id)
    if not state or state.get('df_individual') is None:
        raise HTTPException(404, "No results found")

    # Create Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if state['df_individual'] is not None:
            state['df_individual'].to_excel(writer, sheet_name="All Tickets", index=False)
        if state['df_report'] is not None:
            state['df_report'].to_excel(writer, sheet_name="Cluster Report", index=False)

    output.seek(0)
    headers = {'Content-Disposition': f'attachment; filename="Analysis_{session_id}.xlsx"'}
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.get("/analysis_status/{session_id}", dependencies=[Depends(get_api_key)])
async def get_analysis_status(session_id: str, user_hash: str = Depends(get_user_email_hash)):
    """Polls the status of a background analysis task."""
    # Verify ownership
    if not session_manager.verify_ownership(session_id, user_hash):
        raise HTTPException(403, "Access denied")

    task_info = background.get_task_status(session_id)

    if task_info is None:
        # No in-memory task — check the saved session state
        state = session_manager.load_session(session_id)
        if state and state.get("analysis_complete"):
            return {"status": "completed", "error": None}

        # Detect stale state: analysis_in_progress=True but no background task running
        # This happens when the backend was restarted mid-analysis
        if state and state.get("analysis_in_progress"):
            # Reset the stale flag in Postgres
            state["analysis_in_progress"] = False
            session_manager.save_session(session_id, state)
            logger.warning(f"Reset stale analysis_in_progress for session {session_id} (likely lost due to backend restart)")
            return {"status": "failed", "error": "Analysis was interrupted by a server restart. Please try again."}

        return {"status": "not_started", "error": None}

    return {"status": task_info["status"], "error": task_info.get("error")}

@app.get("/sessions", dependencies=[Depends(get_api_key)])
async def get_sessions(user_hash: str = Depends(get_user_email_hash)):
    """Returns list of previous sessions for the current user."""
    try:
        sessions = session_manager.list_sessions(user_hash)
        return {"sessions" : sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(500, "An internal server error occurred while processing your request. Please try again later.")

@app.patch("/session/{session_id}/title", dependencies=[Depends(get_api_key)])
async def update_session_title_endpoint(session_id: str, req: TitleUpdateRequest, user_hash: str = Depends(get_user_email_hash)):
    """Renames a session."""
    try:
        session_manager.update_session_title(session_id, req.title, user_hash)
        return {"status": "success", "message": "Title updated"}
    except Exception as e:
        logger.error(f"Failed to update title: {e}")
        raise HTTPException(500, "An internal server error occurred while processing your request. Please try again later.")

@app.delete("/session/{session_id}", dependencies=[Depends(get_api_key)])
async def delete_session_endpoint(session_id: str, user_hash: str = Depends(get_user_email_hash)):
    """Deletes a session and cancels any running background analysis."""
    try:
        # Cancel any running background analysis first to save tokens and compute
        background.cancel_analysis_task(session_id)
        session_manager.delete_session(session_id, user_hash)
        return {"status": "success", "message": "Session deleted"}
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(500, "An internal server error occurred while processing your request. Please try again later.")
