"""
Lazy-loading tool documentation for reduced context window usage.

When DATABRICKS_MCP_TOOL_DOCS_MODE=minimal, tool descriptions in tools/list
are replaced with one-liners. A get_tool_docs tool serves full documentation
on-demand.

Default behavior (env var unset or "full") is unchanged.
"""

import logging
import os
from collections.abc import Sequence

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool

logger = logging.getLogger(__name__)

# Descriptions shorter than this are already concise — no benefit to truncating.
_MIN_DESCRIPTION_LENGTH = 300

_HINT_SUFFIX = " (Use get_tool_docs for full details.)"


def _extract_one_liner(description: str) -> str:
    """Extract the first non-empty line from a docstring."""
    for line in description.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


class LazyDocsMiddleware(Middleware):
    """Replaces long tool descriptions with one-liners in tools/list responses."""

    def __init__(self, full_docs: dict[str, str]):
        self._full_docs = full_docs

    async def on_list_tools(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Sequence[Tool]:
        tools = await call_next(context)
        return [self._maybe_minimize(tool) for tool in tools]

    def _maybe_minimize(self, tool: Tool) -> Tool:
        if tool.name == "get_tool_docs":
            return tool
        desc = tool.description
        if not desc or len(desc) <= _MIN_DESCRIPTION_LENGTH:
            return tool
        one_liner = _extract_one_liner(desc)
        return tool.model_copy(update={"description": one_liner + _HINT_SUFFIX})


def _make_get_tool_docs(full_docs: dict[str, str]):
    """Factory that creates the get_tool_docs function closing over full_docs."""

    def get_tool_docs(tool_name: str = None) -> dict:
        """Get full documentation for a Databricks MCP tool.

        When running in minimal docs mode, tool descriptions are abbreviated.
        Use this tool to retrieve the complete documentation for any tool.

        Args:
            tool_name: Name of the tool to get docs for. If omitted, returns
                a catalog of all tools with one-line summaries.

        Returns:
            Full documentation for the requested tool, or a catalog listing.
        """
        if tool_name is None:
            catalog = []
            for name in sorted(full_docs):
                doc = full_docs[name]
                catalog.append({
                    "tool_name": name,
                    "summary": _extract_one_liner(doc),
                    "doc_length": len(doc),
                })
            return {"tools": catalog, "count": len(catalog)}

        if tool_name not in full_docs:
            return {
                "error": f"Unknown tool: {tool_name}",
                "available_tools": sorted(full_docs.keys()),
            }

        return {"tool_name": tool_name, "documentation": full_docs[tool_name]}

    return get_tool_docs


def setup_lazy_docs(mcp_server) -> bool:
    """Configure lazy documentation loading if DATABRICKS_MCP_TOOL_DOCS_MODE=minimal.

    Call this after all tools are registered. In minimal mode:
    1. Snapshots all tool descriptions
    2. Registers a get_tool_docs tool for on-demand lookup
    3. Adds middleware to abbreviate long descriptions in tools/list

    Returns True if minimal mode was activated, False otherwise.
    """
    mode = os.environ.get("DATABRICKS_MCP_TOOL_DOCS_MODE", "full").lower()
    if mode != "minimal":
        return False

    # Snapshot tool descriptions from the internal registry
    try:
        tools = mcp_server._tool_manager._tools
    except AttributeError:
        logger.error(
            "Cannot access tool registry — FastMCP internals may have changed. "
            "Lazy docs disabled."
        )
        return False

    if not tools:
        logger.warning("No tools registered — lazy docs has nothing to snapshot.")
        return False

    full_docs = {
        name: tool.description
        for name, tool in tools.items()
        if tool.description
    }

    logger.info(
        "Lazy docs: captured %d tool descriptions (minimal mode active)", len(full_docs)
    )

    # Register the get_tool_docs tool
    get_tool_docs_fn = _make_get_tool_docs(full_docs)
    mcp_server.tool(get_tool_docs_fn)

    # Add middleware to abbreviate descriptions
    mcp_server.add_middleware(LazyDocsMiddleware(full_docs))

    return True
