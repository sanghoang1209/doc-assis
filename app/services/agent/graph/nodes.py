import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.client import groq_client
from app.services.agent.tools import TOOLS, execute_tool
from app.schemas import AgentState, ChatMessage, Node, ThoughtStep, ToolCallDetail


async def think_node(state: AgentState) -> tuple[AgentState, Node]:
    """Execute LLM reasoning turn to determine whether to call tools or finish answering.

    Args:
        state (AgentState): Current graph state containing conversation messages and step history.

    Returns:
        tuple[AgentState, Node]: Updated agent state and next graph node (EXECUTE or END).
    """
    try:
        response = await groq_client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=state.messages,
            tools=TOOLS,
        )
    except Exception as e:
        raise RuntimeError(f"Generate failed: {e}")

    message = response.choices[0].message
    turn_tokens = response.usage.total_tokens
    new_state = AgentState(
        question=state.question,
        messages=state.messages + [message],
        thought_steps=state.thought_steps,
        last_turn_tokens=turn_tokens,
        loop_count=state.loop_count + 1,
        final_response=state.final_response
    )

    if message.tool_calls:
        return new_state, Node.EXECUTE

    final_step = ThoughtStep(
        loop_index=new_state.loop_count - 1,
        token=turn_tokens,
        thought=message.content,
        tool_calls=[]
    )

    final_state = AgentState(
        question=new_state.question,
        messages=new_state.messages,
        thought_steps=new_state.thought_steps + [final_step],
        last_turn_tokens=new_state.last_turn_tokens,
        loop_count=new_state.loop_count,
        final_response=message.content or "",
    )
    return final_state, Node.END


async def execute_node(state: AgentState, db: AsyncSession) -> tuple[AgentState, Node]:
    """Execute requested tool calls from the last thought turn and record thought steps.

    Args:
        state (AgentState): Current graph state containing the tool call message.
        db (AsyncSession): Database session required for tool executions.

    Returns:
        tuple[AgentState, Node]: Updated agent state with tool execution results and next node (THINK).
    """
    tc_messages = state.messages[-1]
    tool_calls_detail: list[ToolCallDetail] = []
    new_messages: list[dict] = list(state.messages)

    tool_calls = tc_messages.get("tool_calls", []) if isinstance(tc_messages, dict) else (tc_messages.tool_calls or [])

    for tc in tool_calls:
        tc_id = tc["id"] if isinstance(tc, dict) else tc.id
        func = tc["function"] if isinstance(tc, dict) else tc.function
        tool_name = func["name"] if isinstance(func, dict) else func.name
        args_str = func["arguments"] if isinstance(func, dict) else func.arguments
        
        tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str

        result = await execute_tool(
            tool_name=tool_name,
            tool_input=tool_input,
            db=db
        )

        tool_calls_detail.append(ToolCallDetail(
            id=tc_id,
            name=tool_name,
            arguments=tool_input,
            result=str(result)
        ))

        new_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": str(result)
        })

    thought_content = tc_messages.get("content") if isinstance(tc_messages, dict) else tc_messages.content

    new_thought_step = ThoughtStep(
        loop_index=state.loop_count - 1,
        token=state.last_turn_tokens,
        thought=thought_content,
        tool_calls=tool_calls_detail
    )

    new_state = AgentState(
        question=state.question,
        messages=new_messages,
        thought_steps=state.thought_steps + [new_thought_step],
        loop_count=state.loop_count,
        final_response=None
    )

    return new_state, Node.THINK