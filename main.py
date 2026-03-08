import asyncio
import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from agents.models import _openai_shared
from uipath_openai_agents.chat import UiPathChatOpenAI
from uipath_openai_agents.chat.supported_models import OpenAIModels
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()


@asynccontextmanager
async def get_mcp_session():
    MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
    UIPATH_PAT = os.getenv("UIPATH_PAT")

    if not MCP_SERVER_URL:
        raise ValueError("MCP_SERVER_URL must be set")
    if not UIPATH_PAT:
        raise ValueError("UIPATH_PAT must be set")

    headers = {"Authorization": f"Bearer {UIPATH_PAT}"}

    async with streamablehttp_client(
        url=MCP_SERVER_URL,
        headers=headers,
        timeout=60,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@function_tool
async def firecrawl_search(query: str, limit: int = 5) -> str:
    """Search the web using Firecrawl."""
    async with get_mcp_session() as session:
        result = await session.call_tool(
            "firecrawl_search",
            {"query": query, "limit": limit},
        )
        return str(result)


@function_tool
async def firecrawl_scrape(url: str) -> str:
    """Scrape a single webpage."""
    async with get_mcp_session() as session:
        result = await session.call_tool(
            "firecrawl_scrape",
            {"url": url},
        )
        return str(result)


@function_tool
async def firecrawl_crawl(url: str, limit: int = 10) -> str:
    """Crawl an entire website."""
    async with get_mcp_session() as session:
        result = await session.call_tool(
            "firecrawl_crawl",
            {"url": url, "limit": limit},
        )
        return str(result)


@function_tool
async def firecrawl_extract(url: str, extract_schema: str) -> str:
    """Extract structured data using a schema (pass schema as JSON string)."""
    import json
    async with get_mcp_session() as session:
        schema = json.loads(extract_schema)
        result = await session.call_tool(
            "firecrawl_extract",
            {"url": url, "schema": schema},
        )
        return str(result)


MODEL = OpenAIModels.gpt_5_1_2025_11_13

uipath_openai_client = UiPathChatOpenAI(model_name=MODEL)
_openai_shared.set_default_openai_client(uipath_openai_client.async_client)

firecrawl_agent = Agent(
    name="firecrawl_agent",
    instructions="""
You are an advanced web research assistant powered by Firecrawl.

Use the available tools appropriately:
- firecrawl_search: Search the web for information
- firecrawl_scrape: Extract content from a single URL
- firecrawl_crawl: Crawl an entire website
- firecrawl_extract: Extract structured data using a schema

Always use tools to gather information instead of guessing.
Summarize results clearly and concisely.
""",
    model=MODEL,
    tools=[
        firecrawl_search,
        firecrawl_scrape,
        firecrawl_crawl,
        firecrawl_extract,
    ],
)


def agent() -> Agent:
    """UiPath entry point — returns the agent instance."""
    return firecrawl_agent


async def main():
    """Local test entry point."""
    # Read query from CLI arg or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "Search for the latest information about AI and summarize the findings."
    result = await Runner.run(firecrawl_agent, query)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())