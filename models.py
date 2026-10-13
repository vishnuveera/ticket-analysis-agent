from typing import List, Dict, Any, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ContentPayload(BaseModel):
    text: str
    data: Optional[List[Dict[str, Any]]] = None
    charts: Optional[List[Dict[str, Any]]] = None

class ApiResponse(BaseModel):
    session_id: str
    status: str
    content: ContentPayload
    execution_logs: List[str] = []
    session_title: Optional[str] = None

class TitleUpdateRequest(BaseModel):
    title: str

# --- Internal Graph State ---

class AgentState(TypedDict):
    # Standard LangGraph message history
    messages: Annotated[list, add_messages]
    
    # File Context
    file_path: Optional[str]
    session_title: Optional[str]
    
    # Analysis Results
    df_individual: Optional[Any] 
    df_report: Optional[Any] 
    executive_summary: Optional[str]
    logs : Optional[List[str]]
    
    # SQL / Database Context
    db_engine: Any # SQLAlchemy Engine (Runtime only - removed before saving)
    db_file_path: Optional[str] # Path to local sqlite file
    modified_df: Optional[Any]
    chart_data: Optional[Any]  # Plotly chart spec(s) for current response
    
    analysis_complete: bool
    analysis_in_progress: Optional[bool]

    # Runtime-only fields (injected by /chat, stripped before saving to Postgres)
    _session_id: Optional[str]
    _user_email_hash: Optional[str]


class SheetSchema(BaseModel):
    """The identified schema for ticket data."""
    sheet_name: str = Field(description="The name of the Excel sheet containing the ticket data.")
    id_column: str = Field(description="The column header that contains the primary ticket ID (e.g., 'INC12345').")
    summary_column: str = Field(description="The column header that contains the user's ticket description or summary.")
    resolution_column: Optional[str] = Field(None, description="The column header containing resolution notes or technician comments.")

class UsecaseInfo(BaseModel):
    """The generated name for a cluster."""
    usecase_name: str = Field(description="A concise, specific, and technical title for the ticket cluster (e.g., 'Azure Web App SSL Certificate Expiry').")
    technical_description: str = Field(description="A dense, keyword-rich technical summary of the problem scenario, specifically designed for vector matching against solution documents.")

class NoiseAudit(BaseModel):
    noise_ids: List[int]