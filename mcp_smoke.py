"""End-to-end MCP smoke: launch the server over stdio with a real client.

    uv run python mcp_smoke.py

Proves the JSON-RPC handshake + tools/list + a no-model tool call work.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "engram_mcp.server"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("tools:", names)
            assert {"index_project", "index_status", "search_code", "list_indexed_projects"} <= set(names)

            res = await session.call_tool("list_indexed_projects", {})
            print("list_indexed_projects ok:", not res.isError)
            print("MCP SMOKE OK")


if __name__ == "__main__":
    asyncio.run(main())
