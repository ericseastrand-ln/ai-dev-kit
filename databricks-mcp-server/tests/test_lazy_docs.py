"""Tests for lazy-loading tool documentation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_mcp_server.lazy_docs import (
    LazyDocsMiddleware,
    _MIN_DESCRIPTION_LENGTH,
    _extract_one_liner,
    _make_get_tool_docs,
    setup_lazy_docs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name="test_tool", description="Short description."):
    """Build a minimal Tool mock with model_copy support."""
    tool = MagicMock()
    tool.name = name
    tool.description = description

    def model_copy(update=None):
        new = MagicMock()
        new.name = name
        new.description = update.get("description", description) if update else description
        new.model_copy = model_copy
        return new

    tool.model_copy = model_copy
    return tool


def _long_description(first_line="Do something important.", total_length=400):
    """Build a multi-line description that exceeds the minimum length."""
    padding = "\n" + "x" * (total_length - len(first_line) - 1)
    return first_line + padding


# ===========================================================================
# TestExtractOneLiner
# ===========================================================================


class TestExtractOneLiner:
    def test_single_line(self):
        assert _extract_one_liner("Hello world.") == "Hello world."

    def test_multiline_returns_first(self):
        text = "First line.\nSecond line.\nThird line."
        assert _extract_one_liner(text) == "First line."

    def test_leading_whitespace_stripped(self):
        text = "  \n  \n   Real first line.\nOther."
        assert _extract_one_liner(text) == "Real first line."

    def test_empty_string(self):
        assert _extract_one_liner("") == ""

    def test_all_blank_lines(self):
        assert _extract_one_liner("  \n  \n  ") == ""


# ===========================================================================
# TestLazyDocsMiddleware
# ===========================================================================


class TestLazyDocsMiddleware:
    @pytest.fixture
    def full_docs(self):
        return {
            "short_tool": "Short.",
            "long_tool": _long_description("Long tool summary."),
            "get_tool_docs": _long_description("Get full documentation."),
        }

    @pytest.fixture
    def middleware(self, full_docs):
        return LazyDocsMiddleware(full_docs)

    @pytest.mark.asyncio
    async def test_short_description_unchanged(self, middleware):
        """Descriptions under the threshold pass through as-is."""
        tool = _make_tool("short_tool", "Short.")
        call_next = AsyncMock(return_value=[tool])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert len(result) == 1
        assert result[0] is tool  # same object, not a copy

    @pytest.mark.asyncio
    async def test_long_description_minimized(self, middleware):
        """Long descriptions are replaced with one-liner + hint."""
        long_desc = _long_description("Do something complex.")
        tool = _make_tool("long_tool", long_desc)
        call_next = AsyncMock(return_value=[tool])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert len(result) == 1
        minimized = result[0]
        assert minimized is not tool  # new object
        assert minimized.description.startswith("Do something complex.")
        assert "get_tool_docs" in minimized.description

    @pytest.mark.asyncio
    async def test_get_tool_docs_excluded(self, middleware):
        """The get_tool_docs tool is never minimized, even if long."""
        long_desc = _long_description("Get full documentation.")
        tool = _make_tool("get_tool_docs", long_desc)
        call_next = AsyncMock(return_value=[tool])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert result[0] is tool

    @pytest.mark.asyncio
    async def test_none_description_unchanged(self, middleware):
        """Tools with None description pass through."""
        tool = _make_tool("null_tool", None)
        call_next = AsyncMock(return_value=[tool])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert result[0] is tool

    @pytest.mark.asyncio
    async def test_empty_description_unchanged(self, middleware):
        """Tools with empty string description pass through."""
        tool = _make_tool("empty_tool", "")
        call_next = AsyncMock(return_value=[tool])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert result[0] is tool

    @pytest.mark.asyncio
    async def test_mixed_tool_list(self, middleware):
        """Correctly handles a mix of short and long tools."""
        short = _make_tool("short", "Brief.")
        long = _make_tool("long", _long_description("Complex operation."))
        call_next = AsyncMock(return_value=[short, long])
        ctx = MagicMock()

        result = await middleware.on_list_tools(ctx, call_next)

        assert result[0] is short
        assert result[1] is not long
        assert "get_tool_docs" in result[1].description

    @pytest.mark.asyncio
    async def test_call_next_awaited(self, middleware):
        """call_next is properly awaited."""
        call_next = AsyncMock(return_value=[])
        ctx = MagicMock()

        await middleware.on_list_tools(ctx, call_next)

        call_next.assert_awaited_once_with(ctx)


# ===========================================================================
# TestGetToolDocs
# ===========================================================================


class TestGetToolDocs:
    @pytest.fixture
    def full_docs(self):
        return {
            "execute_sql": "Execute a SQL query on a Databricks SQL Warehouse.\n\nLots of details...",
            "list_warehouses": "List all SQL warehouses in the workspace.",
            "create_or_update_dashboard": "Create dashboard.\n\nVery long description " + "x" * 500,
        }

    @pytest.fixture
    def get_tool_docs(self, full_docs):
        return _make_get_tool_docs(full_docs)

    def test_specific_tool_returns_full_doc(self, get_tool_docs, full_docs):
        result = get_tool_docs(tool_name="execute_sql")
        assert result["tool_name"] == "execute_sql"
        assert result["documentation"] == full_docs["execute_sql"]

    def test_unknown_tool_returns_error(self, get_tool_docs):
        result = get_tool_docs(tool_name="nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]
        assert "available_tools" in result
        assert "execute_sql" in result["available_tools"]

    def test_catalog_mode_returns_all_tools(self, get_tool_docs, full_docs):
        result = get_tool_docs()
        assert result["count"] == len(full_docs)
        names = [t["tool_name"] for t in result["tools"]]
        assert sorted(names) == sorted(full_docs.keys())

    def test_catalog_includes_summaries(self, get_tool_docs):
        result = get_tool_docs()
        sql_entry = next(t for t in result["tools"] if t["tool_name"] == "execute_sql")
        assert sql_entry["summary"] == "Execute a SQL query on a Databricks SQL Warehouse."

    def test_catalog_includes_doc_lengths(self, get_tool_docs, full_docs):
        result = get_tool_docs()
        sql_entry = next(t for t in result["tools"] if t["tool_name"] == "execute_sql")
        assert sql_entry["doc_length"] == len(full_docs["execute_sql"])


# ===========================================================================
# TestSetupLazyDocs
# ===========================================================================


class TestSetupLazyDocs:
    def _make_mcp_server(self, tools=None):
        """Build a mock FastMCP server whose list_tools() returns the given tools."""
        server = MagicMock()
        tool_list = []
        if tools:
            for name, t in tools.items():
                t.name = name
                tool_list.append(t)
        server.list_tools = AsyncMock(return_value=tool_list)
        server.tool = MagicMock()
        server.add_middleware = MagicMock()
        return server

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_minimal_mode_activates(self):
        tool = MagicMock()
        tool.description = _long_description("A tool.")
        server = self._make_mcp_server({"my_tool": tool})

        result = setup_lazy_docs(server)

        assert result is True
        server.tool.assert_called_once()
        server.add_middleware.assert_called_once()

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "full"})
    def test_full_mode_is_noop(self):
        server = self._make_mcp_server({"my_tool": MagicMock(description="x")})

        result = setup_lazy_docs(server)

        assert result is False
        server.tool.assert_not_called()
        server.add_middleware.assert_not_called()

    @patch.dict("os.environ", {}, clear=False)
    def test_unset_defaults_to_full(self):
        # Remove the key if present
        import os
        os.environ.pop("DATABRICKS_MCP_TOOL_DOCS_MODE", None)

        server = self._make_mcp_server({"my_tool": MagicMock(description="x")})

        result = setup_lazy_docs(server)

        assert result is False

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_empty_tools_returns_false(self):
        server = self._make_mcp_server({})

        result = setup_lazy_docs(server)

        assert result is False
        server.tool.assert_not_called()

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_middleware_receives_full_docs(self):
        tool = MagicMock()
        tool.description = "The full tool documentation."
        server = self._make_mcp_server({"my_tool": tool})

        setup_lazy_docs(server)

        middleware = server.add_middleware.call_args[0][0]
        assert isinstance(middleware, LazyDocsMiddleware)
        assert middleware._full_docs == {"my_tool": "The full tool documentation."}

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_registered_function_is_get_tool_docs(self):
        tool = MagicMock()
        tool.description = "Some docs."
        server = self._make_mcp_server({"my_tool": tool})

        setup_lazy_docs(server)

        registered_fn = server.tool.call_args[0][0]
        assert registered_fn.__name__ == "get_tool_docs"

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_list_tools_called_with_run_middleware_false(self):
        """Snapshot must skip middleware to capture raw descriptions."""
        tool = MagicMock()
        tool.description = "Docs."
        server = self._make_mcp_server({"my_tool": tool})

        setup_lazy_docs(server)

        server.list_tools.assert_awaited_once_with(run_middleware=False)

    @patch.dict("os.environ", {"DATABRICKS_MCP_TOOL_DOCS_MODE": "minimal"})
    def test_list_tools_failure_returns_false(self):
        """If list_tools raises, lazy docs is disabled rather than crashing server startup."""
        server = MagicMock()
        server.list_tools = AsyncMock(side_effect=RuntimeError("boom"))
        server.tool = MagicMock()
        server.add_middleware = MagicMock()

        result = setup_lazy_docs(server)

        assert result is False
        server.tool.assert_not_called()
        server.add_middleware.assert_not_called()
