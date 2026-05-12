"""
Strip tool input schemas to the bare minimum for reduced context-window cost.

When DATABRICKS_MCP_SCHEMA_MODE=minimal, every advertised tool's ``inputSchema``
is replaced with ``{"type": "object", "additionalProperties": true}``.

Rationale:
    - The server validates arguments server-side via pydantic — the advertised
      schema is primarily a teaching signal for the LLM, not a validation
      contract enforced by the MCP host.
    - ``get_tool_docs`` is already registered by ``setup_lazy_docs`` as an
      on-demand escape hatch exposing the full original docstrings. When the
      model needs a tool's signature, it calls ``get_tool_docs(tool_name=X)``.
    - Parameter names, types, defaults, and nested structure in the advertised
      schema are load-bearing for the model's first-try accuracy but relatively
      expensive in tokens. Pushing that detail behind ``get_tool_docs`` trades
      one round-trip on novel invocations for a cheaper always-loaded surface.

Exceptions:
    ``get_tool_docs`` keeps its real schema — it's the bootstrap tool that
    unlocks every other tool's signature. Stripping its schema would make
    the escape hatch impossible to call.

Default (env var unset or "full") is a no-op passthrough.
"""

import logging
import os
from collections.abc import Sequence

from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool

logger = logging.getLogger(__name__)

# Always-callable tools that need their real schema to bootstrap the system.
_PROTECTED = {"get_tool_docs"}

# Minimal valid JSON Schema for an object-shaped argument blob. Accepts any
# keys (the server's pydantic layer handles actual validation).
_MINIMAL_SCHEMA: dict = {"type": "object", "additionalProperties": True}


class MinimalSchemaMiddleware(Middleware):
    """Replaces inputSchema on tools/list with an empty object schema."""

    async def on_list_tools(
        self,
        context: MiddlewareContext,
        call_next: CallNext,
    ) -> Sequence[Tool]:
        tools = await call_next(context)
        return [self._maybe_strip(tool) for tool in tools]

    def _maybe_strip(self, tool: Tool) -> Tool:
        if tool.name in _PROTECTED:
            return tool
        # Return a fresh dict each time so downstream consumers can't mutate
        # the shared constant.
        return tool.model_copy(update={"parameters": dict(_MINIMAL_SCHEMA)})


def setup_minimal_schema(mcp_server) -> bool:
    """Install MinimalSchemaMiddleware if DATABRICKS_MCP_SCHEMA_MODE=minimal.

    Call this after all tools are registered and after setup_lazy_docs, so
    that get_tool_docs is present and its full schema stays intact.

    Returns True if minimal mode was activated, False otherwise.
    """
    mode = os.environ.get("DATABRICKS_MCP_SCHEMA_MODE", "full").lower()
    if mode != "minimal":
        return False

    mcp_server.add_middleware(MinimalSchemaMiddleware())
    logger.info("Minimal schema mode active: tool inputSchemas will be stripped")
    return True
