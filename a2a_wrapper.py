import os
import time
import logging
import base64
import io
import shutil
from uuid import uuid4
import pandas as pd
import numpy as np

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCard, AgentCapabilities, AgentSkill,
    Message, Part, TextPart, DataPart,
    FilePart, FileWithBytes, Role,
)
from a2a.server.tasks import InMemoryTaskStore
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import internal graph and session helpers
from agent import build_graph
from session import rehydrate_sqlite_from_state
from models import AgentState
from config import UPLOAD_DIR
import background

logger = logging.getLogger("TicketA2AWrapper")

# Configurable Agent Card URL (defaults to localhost for dev)
A2A_AGENT_URL = os.environ.get("A2A_AGENT_URL", "http://localhost:8001/a2a/")

# --- 1. Agent Card Definition ---
ticket_agent_card = AgentCard(
    name="IT Ticket Analysis Agent",
    description=(
        "Analyzes uploaded IT tickets, clusters them using HDBSCAN, "
        "identifies noise, and generates Plotly charts and business summaries."
    ),
    url=A2A_AGENT_URL,
    version="1.0.0",
    default_input_modes=["text", "application/json"],
    default_output_modes=["text", "application/json"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="chat",
            name="Query Ticket Analysis",
            description=(
                "Answer questions about the analyzed IT tickets, generate charts, "
                "usecases and their efforts required and query the SQLite database "
                "for a specific session."
            ),
            tags=["tickets", "analysis", "sql", "charts", "usecases", "efforts"],
            examples=[
                "Show me the top 5 ticket usecases",
                "Generate a pie chart of tickets by type",
            ],
        ),
        AgentSkill(
            id="upload",
            name="Upload File",
            description=(
                "Upload a CSV/Excel file for ticket analysis. "
                "Send the file as a FilePart with base64-encoded bytes."
            ),
            tags=["upload", "file", "csv", "excel"],
            examples=["Upload tickets.xlsx for analysis"],
        ),
        AgentSkill(
            id="analysis_status",
            name="Check Analysis Status",
            description="Poll the status of a background ticket analysis task.",
            tags=["status", "analysis", "poll"],
            examples=["Check if the analysis is complete"],
        ),
        AgentSkill(
            id="download_workbook",
            name="Download Workbook",
            description=(
                "Download the clustered analysis results as an Excel workbook. "
                "Returns a FilePart with base64-encoded .xlsx file."
            ),
            tags=["download", "excel", "workbook", "export"],
            examples=["Download the analysis results"],
        ),
        AgentSkill(
            id="history",
            name="Get Chat History",
            description="Retrieve full chat and data history for a session.",
            tags=["history", "messages", "conversation"],
            examples=["Get the conversation history for this session"],
        ),
        AgentSkill(
            id="get_sessions",
            name="List Sessions",
            description="List all previous sessions for the user.",
            tags=["sessions", "list"],
            examples=["Show me all my previous sessions"],
        ),
        AgentSkill(
            id="update_title",
            name="Update Session Title",
            description="Rename a session's title.",
            tags=["session", "rename", "title"],
            examples=["Rename this session to 'Q4 Analysis'"],
        ),
        AgentSkill(
            id="delete_session",
            name="Delete Session",
            description="Permanently delete a session and its associated data.",
            tags=["session", "delete"],
            examples=["Delete this session"],
        ),
    ],
)


# --- 2. Helper: Extract metadata safely ---
def _get_metadata(message: Message, key: str, default=None):
    """Safely extract a value from message metadata."""
    if message.metadata is None:
        return default
    return message.metadata.get(key, default)


# --- 3. A2A Executor Class ---
class TicketA2AExecutor(AgentExecutor):
    _session_manager = None

    @classmethod
    def set_session_manager(cls, sm):
        cls._session_manager = sm

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Main A2A executor — routes actions to internal handlers."""
        start_time = time.time()
        message: Message = context.message

        # Extract routing context from message metadata
        session_id = _get_metadata(message, "session_id")
        user_email_hash = _get_metadata(message, "user_hash")
        action = _get_metadata(message, "action", "chat")

        if not session_id or not user_email_hash:
            await self._enqueue_error(
                event_queue, "Missing 'session_id' or 'user_hash' in message metadata.",
                400, "MISSING_CONTEXT"
            )
            return

        try:
            # ==========================================
            # ROUTE: UPLOAD FILE
            # ==========================================
            if action == "upload":
                await self._handle_upload(event_queue, message, session_id, user_email_hash)

            # ==========================================
            # ROUTE: ANALYSIS STATUS (POLLING)
            # ==========================================
            elif action == "analysis_status":
                await self._handle_analysis_status(event_queue, session_id, user_email_hash)

            # ==========================================
            # ROUTE: DOWNLOAD WORKBOOK
            # ==========================================
            elif action == "download_workbook":
                await self._handle_download_workbook(event_queue, session_id, user_email_hash)

            # ==========================================
            # ROUTE: CHAT HISTORY
            # ==========================================
            elif action == "history":
                await self._handle_history(event_queue, session_id, user_email_hash)

            # ==========================================
            # ROUTE: LIST SESSIONS
            # ==========================================
            elif action == "get_sessions":
                sessions = self._session_manager.list_sessions(user_email_hash)
                await self._enqueue_success(event_queue, "Sessions retrieved.", {"sessions": sessions})

            # ==========================================
            # ROUTE: UPDATE SESSION TITLE
            # ==========================================
            elif action == "update_title":
                new_title = _get_metadata(message, "title")
                if not new_title:
                    await self._enqueue_error(event_queue, "Missing 'title' in metadata.", 400, "MISSING_TITLE")
                    return
                self._session_manager.update_session_title(session_id, new_title, user_email_hash)
                await self._enqueue_success(event_queue, "Title updated.", {"status": "success"})

            # ==========================================
            # ROUTE: DELETE SESSION
            # ==========================================
            elif action == "delete_session":
                background.cancel_analysis_task(session_id)
                self._session_manager.delete_session(session_id, user_email_hash)
                await self._enqueue_success(event_queue, "Session deleted.", {"status": "success"})

            # ==========================================
            # ROUTE: CHAT (Default LangGraph Execution)
            # ==========================================
            elif action == "chat":
                await self._handle_chat(event_queue, message, session_id, user_email_hash, start_time)

            else:
                await self._enqueue_error(
                    event_queue, f"Unknown action: '{action}'", 400, "UNKNOWN_ACTION"
                )

        except Exception as exc:
            logger.error(f"A2A Executor Error: {exc}", exc_info=True)
            await self._enqueue_error(
                event_queue, "Internal Server Error during execution.", 500, "EXEC_500"
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation (not supported for this agent)."""
        await self._enqueue_error(
            event_queue, "Cancellation is not supported.", 400, "CANCEL_NOT_SUPPORTED"
        )

    # =========================================================================
    # HANDLER: UPLOAD
    # =========================================================================
    async def _handle_upload(self, event_queue, message, session_id, user_email_hash):
        """Handle file upload via A2A FilePart."""
        # Find the FilePart in the message
        file_part = None
        for p in message.parts:
            part = p.root if hasattr(p, 'root') else p
            if isinstance(part, FilePart):
                file_part = part
                break

        if not file_part:
            # Fallback: check for DataPart with file_data (backward compat)
            data_part = None
            for p in message.parts:
                part = p.root if hasattr(p, 'root') else p
                if isinstance(part, DataPart) and "file_data" in (part.data or {}):
                    data_part = part
                    break

            if data_part:
                file_data = base64.b64decode(data_part.data["file_data"])
                filename = _get_metadata(message, "filename", "uploaded_file.xlsx")
            else:
                await self._enqueue_error(
                    event_queue, "No FilePart or file data found in message.", 400, "MISSING_FILE"
                )
                return
        else:
            # Extract from FilePart (proper A2A way)
            file_obj = file_part.file
            if isinstance(file_obj, FileWithBytes):
                file_data = base64.b64decode(file_obj.bytes)
                filename = file_obj.name or _get_metadata(message, "filename", "uploaded_file.xlsx")
            else:
                # FileWithUri — not supported (would need to download from URI)
                await self._enqueue_error(
                    event_queue, "FileWithUri is not supported. Please send file bytes directly.",
                    400, "URI_NOT_SUPPORTED"
                )
                return

        # Save file to disk
        fpath = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
        with open(fpath, "wb") as f:
            f.write(file_data)

        clean_title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()

        # Deduplicate session title (e.g., "Course" → "Course 1" if "Course" exists)
        existing_sessions = self._session_manager.list_sessions(user_email_hash)
        existing_titles = {s.get("title") for s in existing_sessions}
        final_title = clean_title
        suffix_counter = 1
        while final_title in existing_titles:
            final_title = f"{clean_title} {suffix_counter}"
            suffix_counter += 1

        # Initialize default state (same as REST endpoint)
        initial_state = AgentState(
            messages=[], file_path=fpath, df_individual=None, df_report=None,
            executive_summary=None, db_engine=None, modified_df=None, chart_data=None,
            db_file_path=None, analysis_complete=False, analysis_in_progress=False,
            session_title=final_title, logs=[], _session_id=None, _user_email_hash=None
        )
        self._session_manager.save_session(session_id, initial_state, title=final_title, user_email_hash=user_email_hash)

        # Enforce rolling session limit: delete oldest sessions if user exceeds the cap
        self._session_manager.enforce_session_limit(user_email_hash)

        await self._enqueue_success(
            event_queue, "File uploaded and processed.", {"status": "success", "message": "File uploaded and processed."}
        )

    # =========================================================================
    # HANDLER: ANALYSIS STATUS
    # =========================================================================
    async def _handle_analysis_status(self, event_queue, session_id, user_email_hash):
        """Poll analysis status — mirrors REST /analysis_status endpoint."""
        if not self._session_manager.verify_ownership(session_id, user_email_hash):
            await self._enqueue_error(event_queue, "Access denied", 403, "AUTH_403")
            return

        task_info = background.get_task_status(session_id)

        if task_info is None:
            state = self._session_manager.load_session(session_id)
            if state and state.get("analysis_complete"):
                await self._enqueue_success(event_queue, "Completed", {"status": "completed", "error": None})
            elif state and state.get("analysis_in_progress"):
                # Stale state correction (server restart scenario)
                state["analysis_in_progress"] = False
                self._session_manager.save_session(session_id, state)
                logger.warning(f"Reset stale analysis_in_progress for session {session_id}")
                await self._enqueue_success(event_queue, "Failed", {
                    "status": "failed",
                    "error": "Analysis was interrupted by a server restart. Please try again."
                })
            else:
                await self._enqueue_success(event_queue, "Not Started", {"status": "not_started", "error": None})
        else:
            await self._enqueue_success(event_queue, "Status retrieved", {
                "status": task_info["status"],
                "error": task_info.get("error")
            })

    # =========================================================================
    # HANDLER: DOWNLOAD WORKBOOK
    # =========================================================================
    async def _handle_download_workbook(self, event_queue, session_id, user_email_hash):
        """Download analysis results as Excel — returns FilePart with base64 bytes."""
        if not self._session_manager.verify_ownership(session_id, user_email_hash):
            await self._enqueue_error(event_queue, "Access denied", 403, "AUTH_403")
            return

        state = self._session_manager.load_session(session_id)
        if not state or state.get('df_individual') is None:
            await self._enqueue_error(event_queue, "No results found", 404, "NOT_FOUND")
            return

        # Create Excel workbook in memory (same as REST endpoint)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if state['df_individual'] is not None:
                state['df_individual'].to_excel(writer, sheet_name="All Tickets", index=False)
            if state['df_report'] is not None:
                state['df_report'].to_excel(writer, sheet_name="Cluster Report", index=False)

        excel_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
        filename = f"Analysis_{session_id}.xlsx"

        # Return as proper FilePart (A2A protocol compliant)
        response_parts = [
            Part(root=TextPart(text="Workbook generated successfully.")),
            Part(root=FilePart(
                file=FileWithBytes(
                    bytes=excel_b64,
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    name=filename,
                )
            )),
        ]
        await event_queue.enqueue_event(
            Message(role=Role.agent, message_id=str(uuid4()), parts=response_parts)
        )

    # =========================================================================
    # HANDLER: HISTORY
    # =========================================================================
    async def _handle_history(self, event_queue, session_id, user_email_hash):
        """Reconstruct full conversation history — mirrors REST /history endpoint."""
        state = self._session_manager.load_session(session_id)
        if not state:
            await self._enqueue_error(event_queue, "Session not found", 404, "SESSION_404")
            return

        if not self._session_manager.verify_ownership(session_id, user_email_hash):
            await self._enqueue_error(event_queue, "Access denied", 403, "AUTH_403")
            return

        # 1. Recover File Name & Analysis Status
        analysis_complete = state.get("analysis_complete", False)
        file_path = state.get("file_path")
        file_name = None
        if file_path:
            base_name = os.path.basename(file_path)
            parts = base_name.split('_', 1)
            file_name = parts[1] if len(parts) > 1 else base_name

        raw_messages = state.get("messages", [])
        formatted_messages = []

        # 2. Message Reconstruction Loop (identical to REST endpoint)
        buffered_data = None
        buffered_chart = None

        for msg in raw_messages:
            role = "unknown"
            data_payload = None
            chart_payload = None
            content = ""

            if isinstance(msg, HumanMessage):
                if msg.content.startswith("SYSTEM_EVENT"):
                    continue
                role = "user"
                content = msg.content

            elif isinstance(msg, ToolMessage):
                if "dataframe" in msg.additional_kwargs:
                    df = msg.additional_kwargs["dataframe"]
                    if isinstance(df, pd.DataFrame):
                        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("")
                        buffered_data = df_clean.astype(str).to_dict(orient="records")
                if "chart" in msg.additional_kwargs:
                    buffered_chart = msg.additional_kwargs["chart"]
                continue

            elif isinstance(msg, AIMessage):
                if getattr(msg, 'tool_calls', None):
                    continue

                role = "assistant"
                content = msg.content
                if buffered_data:
                    data_payload = buffered_data
                    buffered_data = None
                if buffered_chart:
                    chart_payload = buffered_chart
                    buffered_chart = None
                if "charts" in msg.additional_kwargs:
                    chart_payload = msg.additional_kwargs["charts"]
                if "dataframe" in msg.additional_kwargs:
                    df = msg.additional_kwargs["dataframe"]
                    if isinstance(df, pd.DataFrame):
                        df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("")
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

        # Check if analysis is in progress
        analysis_in_progress = state.get("analysis_in_progress", False)
        if background.is_analysis_running(session_id):
            analysis_in_progress = True

        history_payload = {
            "session_id": session_id,
            "messages": formatted_messages,
            "logs": state.get("logs", []),
            "analysis_complete": analysis_complete,
            "analysis_in_progress": analysis_in_progress,
            "file_name": file_name,
        }

        await self._enqueue_success(event_queue, "History retrieved.", history_payload)

    # =========================================================================
    # HANDLER: CHAT
    # =========================================================================
    async def _handle_chat(self, event_queue, message, session_id, user_email_hash, start_time):
        """Run LangGraph agent — mirrors REST /chat endpoint."""
        # Extract text from A2A message parts
        user_text = ""
        for p in message.parts:
            part = p.root if hasattr(p, 'root') else p
            if isinstance(part, TextPart):
                user_text = part.text
                break

        # Load and verify session
        state = self._session_manager.load_session(session_id)
        if not state:
            await self._enqueue_error(event_queue, "Session not found", 404, "SESSION_404")
            return
        if not self._session_manager.verify_ownership(session_id, user_email_hash):
            await self._enqueue_error(event_queue, "Access denied", 403, "AUTH_403")
            return

        # Hydrate SQLite, reset transient fields (same as REST)
        state = rehydrate_sqlite_from_state(state)
        state['modified_df'] = None
        state['chart_data'] = None
        state['_session_id'] = session_id
        state['_user_email_hash'] = user_email_hash
        state['messages'].append(HumanMessage(content=user_text))

        # Invoke LangGraph
        app_graph = build_graph()
        result = await app_graph.ainvoke(state)

        # Clean and save state (strip runtime-only fields)
        result.pop('_session_id', None)
        result.pop('_user_email_hash', None)
        current_title = result.get("session_title")
        self._session_manager.save_session(session_id, result, title=current_title)

        # Format response payload (same logic as REST /chat)
        last_msg = result['messages'][-1]

        data_payload = None
        if result.get('modified_df') is not None:
            df = result['modified_df']
            if isinstance(df, pd.DataFrame):
                df_clean = df.replace([np.inf, -np.inf], np.nan).fillna("")
                data_payload = df_clean.astype(str).to_dict(orient="records")
                result['modified_df'] = None

        chart_payload = None
        if result.get('chart_data') is not None:
            chart_payload = result['chart_data']
            if not isinstance(chart_payload, list):
                chart_payload = [chart_payload]
            result['chart_data'] = None

        # Save state again after clearing transient fields (mirrors REST double-save)
        if data_payload is not None or chart_payload is not None:
            self._session_manager.save_session(session_id, result, title=current_title)

        # Build response parts
        response_data = {
            "session_id": session_id,
            "status": "success",
            "content": {
                "text": last_msg.content,
                "data": data_payload,
                "charts": chart_payload,
            },
            "execution_logs": result.get("logs", []),
            "session_title": current_title,
        }

        response_parts = [Part(root=TextPart(text=last_msg.content))]
        response_parts.append(Part(root=DataPart(data=response_data)))

        await event_queue.enqueue_event(
            Message(role=Role.agent, message_id=str(uuid4()), parts=response_parts)
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    async def _enqueue_error(self, event_queue, message_text: str, status_code: int, error_code: str):
        """Send an error response via the event queue."""
        error_data = {"error": {"code": error_code, "message": message_text, "status_code": status_code}}
        await event_queue.enqueue_event(
            Message(
                role=Role.agent,
                message_id=str(uuid4()),
                parts=[Part(root=DataPart(data=error_data))]
            )
        )

    async def _enqueue_success(self, event_queue, text_msg: str, data: dict):
        """Send a success response with text + structured data."""
        response_parts = [
            Part(root=TextPart(text=text_msg)),
            Part(root=DataPart(data=data)),
        ]
        await event_queue.enqueue_event(
            Message(role=Role.agent, message_id=str(uuid4()), parts=response_parts)
        )


# --- 4. A2A Application Setup ---
a2a_app = A2AStarletteApplication(
    agent_card=ticket_agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=TicketA2AExecutor(),
        task_store=InMemoryTaskStore(),
    ),
)