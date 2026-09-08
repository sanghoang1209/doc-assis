import logging
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, HTTPException
from app.database import get_db
from app.schemas import AgentQuery
from app.services.agent.loop.run import run_agent_stream
from app.services.agent.graph.core import AgentGraph

logger = logging.getLogger(__name__)

MAX_LOOPS = 8

agent_router = APIRouter(prefix='/agent')

@agent_router.post('/query')
async def chat(
    payload: AgentQuery,
    db: AsyncSession = Depends(get_db),
):
    try:
        agent = AgentGraph(max_loops=MAX_LOOPS)

        return StreamingResponse(
            agent.run(
                question=payload.question,
                chat_history=payload.chat_history,
                document_id=payload.document_id,
                db=db,
            ),
            media_type="text/event-stream"
        )

    except RuntimeError as e:
        logger.error(f"Agent failed to run: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"AI service is temporarily unavailable, Please try again later."
        )