import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.config import PAGE_SIZE
from app.models import Chunk, Document
from app.services.rag.retriever import retrieve_vec


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": (
                "Search for relevant text chunks in a specific document based on the user's query. "
                "Use this tool when the user asks about the content of an uploaded document."
                "Use for targeted lookups; prefer get_full_document only when user explicitly asks to read the entire file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The UUID of the document to search within.",
                    },
                    "query": {
                        "type": "string",
                        "description": "The query or search term to look for in the document.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of relevant text chunks to return. Defaults to 5.",
                        "default": 5,
                    },
                },
                "required": ["document_id", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "List all documents that have been uploaded to the system. "
                "Use this tool when the user wants to know what documents are available "
                "or needs to find a document_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of documents to return. Defaults to 20.",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_full_document",
            "description": (
                "Retrieve the full content of a document page by page. Use page parameter to navigate."
                "Use this tool when you need to read the entire document text instead of "
                "just searching for relevant chunks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The UUID of the document to retrieve the full content for.",
                    },
                    "page": {
                        "type": "integer", 
                        "default": 1, 
                        "description": "Page number to read (each page is ~4000 chars)"
                    }
                },
                "required": ["document_id", "page"],
            },
        },
    },
]

async def execute_tool(
    tool_name: str,
    tool_input: dict,
    db: AsyncSession,
) -> str:
    """
    Execute a tool and return the result as a string.
    LLM only receives strings — format clearly.
    """
    if not db:
        return "Error: No database connection available to execute query."
 
    if tool_name == "search_document":
        document_id_str = tool_input["document_id"]
        query = tool_input["query"]
        top_k = tool_input.get("top_k", 5)

        doc_uuid = None
        doc = None

        if document_id_str:
            try:
                doc_uuid = uuid.UUID(document_id_str)
            except ValueError:
                return f"Error: '{document_id_str}' is not a valid UUID."
 
            # Check if the document exists
            doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
            if not doc:
                return f"Error: Document with ID '{document_id_str}' not found."
 
        rows: list[tuple[Chunk, float]] = await retrieve_vec(query, doc_uuid, top_k, db)
 
        if not rows:
            return "No relevant text chunks found for this query."

        result_lines = []

        if doc_uuid:
            result_lines.append(f"Found {len(rows)} relevant chunks in '{doc.filename}':\n")
            for i, row in enumerate(rows, 1):
                chunk = row[0]
                distance = row[1]
                result_lines.append(f"[Chunk {i} | Distance: {distance}]\n{chunk.content}\n")
        else:
            result_lines.append(f"Found {len(rows)} relevant chunks across all documents:\n")
            for i, row in enumerate(rows, 1):
                chunk = row[0]
                filename = row[1]
                distance = row[2]
                result_lines.append(
                    f"[Chunk {i}] (From Document: '{filename}' | ID: {chunk.document_id} | Distance: {distance})\n"
                    f"{chunk.content}\n"
                )
 
        return "\n".join(result_lines)
 
    elif tool_name == "list_documents":
        limit = tool_input.get("limit", 20)
        docs = (
            await db.scalars(
                select(Document)
                .order_by(Document.created_at.desc())
                .limit(limit)
            )
        ).all()
 
        if not docs:
            return "No documents have been uploaded to the system yet."
 
        lines = [f"The system has {len(docs)} documents:\n"]
        for doc in docs:
            lines.append(
                f"- ID: {doc.id}\n"
                f"  Filename: {doc.filename}\n"
                f"  Uploaded at: {doc.created_at.strftime('%Y-%m-%d %H:%M')}\n"
            )
        return "\n".join(lines)
 
    elif tool_name == "get_full_document":
        document_id_str = tool_input["document_id"]
        page = tool_input.get("page", 1)
        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
 
        try:
            doc_uuid = uuid.UUID(document_id_str)
        except ValueError:
            return f"Error: '{document_id_str}' is not a valid UUID."
 
        doc = await db.scalar(select(Document).where(Document.id == doc_uuid))
        if not doc:
            return f"Error: Document with ID '{document_id_str}' not found."
 
        content_chunk = doc.content[start:end]
        total_pages = (len(doc.content) + PAGE_SIZE - 1) // PAGE_SIZE
 
        return (
            f"=== Document '{doc.filename}' (Page {page}/{total_pages}) ===\n\n"
            f"{content_chunk}\n\n"
            f"[System Note: Page {page} of {total_pages}. "
            f"{'Call get_full_document with page=' + str(page+1) + ' to read more.' if page < total_pages else 'End of document.'}]"
        )
 
    else:
        return f"Error: Tool '{tool_name}' is not recognized."