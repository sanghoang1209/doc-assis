import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.agent.graph.nodes import execute_node, think_node
from app.schemas import (
    AgentState,
    ChatMessage, 
    AgentResponse,
    Node, 
)

class AgentGraph:
    """Graph-based Agent runner controlling state transitions between THINK and EXECUTE nodes."""

    def __init__(self, max_loops: int = 8):
        """Initialize the AgentGraph executor.

        Args:
            max_loops (int, optional): Maximum loop iterations permitted. Defaults to 8.
        """
        self.max_loops = max_loops

    async def run(
        self,
        question: str,
        chat_history: list[ChatMessage] | None,
        document_id: uuid.UUID | None,
        db: AsyncSession,
    ) -> AgentResponse:
        """Execute the agent graph state machine asynchronously.

        Args:
            question (str): User query/question.
            chat_history (list[ChatMessage] | None): Conversation history messages.
            document_id (uuid.UUID | None): Optional specific document ID context.
            db (AsyncSession): Database session for tool execution.

        Returns:
            AgentResponse: Final answer and tracked thought steps.
        """
        initial_messages = self._build_init_message(
            question=question,
            chat_history=chat_history,
            document_id=document_id
        )

        state = AgentState(
            question=question,
            messages=initial_messages
        )

        current_node = Node.THINK

        while current_node != Node.END:
            if state.loop_count >= self.max_loops:
                return AgentResponse(
                    answer=f"Reached max loops ({self.max_loops}) without answer.",
                    thought_steps=state.thought_steps
                )

            if current_node == Node.THINK:
                state, current_node = await think_node(state)

            elif current_node == Node.EXECUTE:
                state, current_node = await execute_node(state, db)

        return AgentResponse(
            answer=state.final_response or "",
            thought_steps=state.thought_steps
        )

    def _build_init_message(
        self,
        question: str,
        chat_history: list[ChatMessage] | None,
        document_id: uuid.UUID | None
    ) -> list[dict]:
        """Construct system prompt and initial message array for the agent graph state.

        Args:
            question (str): User question.
            chat_history (list[ChatMessage] | None): Preceding chat messages.
            document_id (uuid.UUID | None): Target document UUID if scoped.

        Returns:
            list[dict]: List of formatted message dictionaries.
        """
        system_parts = [
            "You are an intelligent assistant that can search for and read document content.",
            "When the user asks about document content, use the tool to search before answering.",
            "Respond in English, concisely, and base your answer on the information found.",
            "If no relevant information is found, state that clearly.",
        ]
    
        if document_id:
            system_parts.append(
                f"\nThe user is asking about the document with ID: {document_id}. "
                f"Prioritize using the search_document tool with this document_id."
            )
        
        system_prompt = "\n".join(system_parts)
    
        messages = [
            {"role": "system", "content": system_prompt},
        ]
    
        if chat_history:
            for chat in chat_history:
                messages.append(chat.model_dump())
    
        messages.append({"role": "user", "content": question})

        return messages