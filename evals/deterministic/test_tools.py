import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.agent.tools import (
    search_document,
    list_documents,
)

@pytest.mark.asyncio
async def test_search_document_with_invalid_uuid():
    db = AsyncMock()
    tool_input = {
        "document_id": "1",
        "query": "haha"
    }

    result = await search_document(tool_input, db)
    assert result == f"Error: '{tool_input['document_id']}' is not a valid UUID."
    db.scalar.assert_not_called()

@pytest.mark.asyncio
async def test_search_document_with_doc_not_found():
    db = AsyncMock()
    db.scalar.return_value = None
    tool_input = {
        "document_id": "431be7c4-efe1-4a8a-b55d-942efab31da1",
        "query": "haha"
    }

    result = await search_document(tool_input, db)
    assert result == f"Error: Document with ID '{tool_input['document_id']}' not found."

@pytest.mark.asyncio
async def test_list_documents_with_no_doc():
    db = AsyncMock()
    scalars_result = MagicMock()
    scalars_result.all.return_value = []
    db.scalars.return_value = scalars_result
    tool_input = {

    }

    result = await list_documents(tool_input, db)
    assert result == "No documents have been uploaded to the system yet."

