import pytest
from app.mcp.server import mcp_server

@pytest.mark.asyncio
async def test_mcp_list_tools():
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    
    assert "search_document" in tool_names
    assert "list_documents" in tool_names
    assert "get_full_document" in tool_names
    assert "summarize_document" in tool_names

@pytest.mark.asyncio
async def test_mcp_call_list_documents():
    res = await mcp_server.call_tool("list_documents", {"limit": 2})
    assert res is not None
    assert hasattr(res, "content")
    assert len(res.content) > 0
    assert "documents" in res.content[0].text or "No documents" in res.content[0].text

@pytest.mark.asyncio
async def test_mcp_call_search_document_invalid_uuid():
    res = await mcp_server.call_tool("search_document", {"query": "test", "document_id": "invalid-uuid"})
    assert res is not None
    assert "not a valid UUID" in res.content[0].text

def test_mcp_sse_app_creation():
    sse_app = mcp_server.sse_app()
    assert sse_app is not None
