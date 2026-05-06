"""Optional tool allowlist to reduce MCP context-window cost.

When DATABRICKS_MCP_TOOLS_ALLOWED is set (comma-separated tool names),
only those tools remain registered. All others are removed at startup.

get_tool_docs is always kept — it's the escape hatch that lets Claude
look up documentation for any tool the server knows about, including
the pruned ones, without the schema cost of listing them.

If the env var is unset or empty, no filtering happens (default behavior).

Ordering note: this must run AFTER setup_lazy_docs so that the get_tool_docs
catalog is built from the full tool set before we prune the registry.
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_PROTECTED = {"get_tool_docs"}


def apply_tool_allowlist(mcp_server) -> None:
    """Remove all tools not in DATABRICKS_MCP_TOOLS_ALLOWED from the registry."""
    raw = os.environ.get("DATABRICKS_MCP_TOOLS_ALLOWED", "").strip()
    if not raw:
        return

    configured = {name.strip() for name in raw.split(",") if name.strip()}
    allowed = configured | _PROTECTED

    try:
        tools = asyncio.run(mcp_server.list_tools(run_middleware=False))
    except Exception as e:
        logger.error("Failed to snapshot tool list for allowlist: %s", e)
        return

    live_names = {t.name for t in tools}
    unknown = sorted(configured - live_names)
    if unknown:
        logger.warning(
            "DATABRICKS_MCP_TOOLS_ALLOWED references unknown tools (ignored): %s",
            ", ".join(unknown),
        )

    provider = mcp_server.local_provider
    removed = []
    for tool in tools:
        if tool.name not in allowed:
            try:
                provider.remove_tool(tool.name)
                removed.append(tool.name)
            except KeyError:
                pass

    logger.info(
        "Tool allowlist active: kept %d, removed %d",
        len(tools) - len(removed),
        len(removed),
    )
