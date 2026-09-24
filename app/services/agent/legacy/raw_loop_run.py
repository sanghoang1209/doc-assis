import json
import uuid
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas import (
    ChatMessage, 
    AgentResponse, 
    ThoughtStep, 
    ToolCallDetail,
)
from app.services.client import groq_client
from app.services.agent.tools import TOOLS, execute_tool


load_dotenv()

async def run_agent(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    db: AsyncSession | None = None,
    document_id: uuid.UUID | None = None,
    max_loops: int = 8
) -> AgentResponse:
    """Run the Agent in non-streaming mode, resolving tool loops internally and returning the final response.

    Args:
        question (str): The query of the user.
        chat_history (list[ChatMessage] | None, optional): Preceding chat messages in the conversation. Defaults to None.
        db (AsyncSession | None, optional): Database session. Defaults to None.
        document_id (uuid.UUID | None, optional): The ID of the document to focus the search context on. Defaults to None.
        max_loops (int, optional): Maximum tool loop iterations. Defaults to 8.

    Returns:
        AgentResponse: The complete structured response containing final answer and thought steps.
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

    thought_steps: list[ThoughtStep] = []

    loops = 0
    while loops < max_loops:
        tool_calls: list[ToolCallDetail] = []
        try:
            response = await groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=messages,
                tools=TOOLS,
            )
        except Exception as e:
            raise RuntimeError(f"Generate failed: {e}")

        messages.append(response.choices[0].message)

        if not response.choices[0].message.tool_calls:
            return AgentResponse(
                answer=response.choices[0].message.content or "",
                thought_steps=thought_steps
            )

        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)

            result = await execute_tool(
                tool_name=tool_name,
                tool_input=tool_input,
                db=db
            )

            tool_calls.append(ToolCallDetail(
                id=tool_call.id,
                name=tool_name,
                arguments=tool_input,
                result=str(result)
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        thought_steps.append(ThoughtStep(
            loop_index=loops,
            token=response.usage.total_tokens,
            thought=response.choices[0].message.content,
            tool_calls=tool_calls
        ))

        loops += 1

    return AgentResponse(
        answer=f"No answer from Agent with tool with {loops} loops\n",
        thought_steps=thought_steps
    )

async def run_agent_stream(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    db: AsyncSession | None = None,
    document_id: uuid.UUID | None = None,
    max_loops: int = 8
):
    """Run the Agent in streaming mode (SSE), yielding intermediate thought events and raw text answer chunks.

    Args:
        question (str): The query of the user.
        chat_history (list[ChatMessage] | None, optional): Preceding chat messages in the conversation. Defaults to None.
        db (AsyncSession | None, optional): Database session. Defaults to None.
        document_id (uuid.UUID | None, optional): The ID of the document to focus the search context on. Defaults to None.
        max_loops (int, optional): Maximum tool loop iterations. Defaults to 8.

    Returns:
        AsyncGenerator[str, None]: Yields SSE formatted strings for 'thought', 'answer', and 'done' events.
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

    loops = 0
    while loops < max_loops:
        tool_calls: list[ToolCallDetail] = []
        accumulated_tool_calls = {}
        accumulated_content = ""
        role = "assistant"
        calling_tools = False
        total_tokens = 0

        try:
            response = await groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b",
                messages=messages,
                tools=TOOLS,
                stream=True
            )
        except Exception as e:
            raise RuntimeError(f"Generate failed: {e}")

        async for chunk in response:
            delta = chunk.choices[0].delta

            if getattr(delta, "role", None) is not None:
                role = delta.role

            if getattr(chunk, "usage", None) is not None:
                total_tokens = chunk.usage.total_tokens

            if getattr(delta, "content", None) is not None:
                piece_content = delta.content
                accumulated_content += piece_content

                yield f"event: answer\ndata: {json.dumps({"text": piece_content})}\n\n"

            if getattr(delta, "tool_calls", None) is not None:
                calling_tools = True
                for tc in delta.tool_calls:
                    index = tc.index

                    if index not in accumulated_tool_calls:
                        accumulated_tool_calls[index] = {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": ""
                        }
 
                    if tc.function.arguments:
                        accumulated_tool_calls[index]["arguments"] += tc.function.arguments

        if not calling_tools:   
            messages.append({
                "role": role,
                "content": accumulated_content
            })

            yield f"event: done\ndata: {json.dumps({'status': 'completed'})}\n\n"
            return

        final_tool_calls = []
        tool_calls_payload = []
        for idx, tc in accumulated_tool_calls.items():
            try:
                parsed_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                parsed_args = {}

            final_tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": parsed_args
            })

            tool_calls_payload.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": tc["arguments"]
                }
            })

        messages.append({
            "role": role,
            "content": accumulated_content,
            "tool_calls": tool_calls_payload
        })
            
        for tc in final_tool_calls:
            result = await execute_tool(
                tool_name=tc["name"],
                tool_input=tc["arguments"],
                db=db
            )

            tool_calls.append(ToolCallDetail(
                id=tc["id"],
                name=tc["name"],
                arguments=tc["arguments"],
                result=str(result)
            ))

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result)
            })

        thought_step=ThoughtStep(
            loop_index=loops,
            token=total_tokens,
            thought=accumulated_content,
            tool_calls=tool_calls
        )
        yield f"event: thought\ndata: {thought_step.model_dump_json()}\n\n"

        loops += 1
        
    yield f"event: error\ndata: {json.dumps({'detail': f'Reached maximum tool loops ({max_loops}) without answer.'})}\n\n"
    yield f"event: done\ndata: {json.dumps({'status': 'max_loops_exceeded'})}\n\n"
