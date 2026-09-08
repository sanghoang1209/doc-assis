from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Literal
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from app.models import FileStatus

class DocumentListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: FileStatus
    filename: str
    updated_at: datetime
    created_at: datetime

class DocumentUploadResponse(DocumentListItem):
    pass

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    token: int
    sources: list[str]

class ToolCallDetail(BaseModel):
    id: str
    name: str
    arguments: dict
    result: str | None = None

class ThoughtStep(BaseModel):
    loop_index: int
    token: int
    thought: str | None = None
    tool_calls: list[ToolCallDetail] = []

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AgentQuery(QueryRequest):
    document_id: uuid.UUID | None = None
    chat_history: list[ChatMessage] | None = None

class AgentResponse(BaseModel):
    answer: str 
    thought_steps: list[ThoughtStep] = []

class Node(str, Enum):
    THINK   = "think"
    EXECUTE = "execute"
    END     = "end"

@dataclass
class AgentState():
    question: str
    messages: list[dict] = field(default_factory=list)
    thought_steps: list[ThoughtStep] = field(default_factory=list)
    last_turn_tokens: int = 0
    loop_count: int = 0
    final_response: str | None = None

@dataclass
class NodeTransition:
    state: AgentState
    next_node: Node