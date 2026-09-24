import sys
from app.mcp.server import mcp_server

def main():
    """Run the Document QA MCP Server using Stdio transport."""
    mcp_server.run(transport="stdio")

if __name__ == "__main__":
    main()
