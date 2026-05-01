"""Tests for MCP integration helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.extended_openai_conversation.exceptions import (
    McpConfigError,
    McpServerError,
)
from custom_components.extended_openai_conversation.mcp import (
    discover_mcp_tools,
    execute_mcp_tool,
    parse_mcp_servers_config,
)


def test_parse_mcp_servers_config_valid() -> None:
    """Test parsing valid MCP server config."""
    result = parse_mcp_servers_config("""
servers:
  - name: docs
    url: https://example.com/mcp
    transport: streamable_http
    allow_tools:
      - search
""")

    assert len(result) == 1
    assert result[0].name == "docs"
    assert result[0].allow_tools == ["search"]


def test_parse_mcp_servers_config_invalid_shape() -> None:
    """Test parsing invalid MCP config shape."""
    with pytest.raises(McpConfigError):
        parse_mcp_servers_config("servers: invalid")


@pytest.mark.asyncio
async def test_discover_mcp_tools_allowlist(hass: MagicMock, monkeypatch) -> None:
    """Test MCP discovery applies allowlist and prefixes tool names."""

    async def mock_json_rpc_call(session, server, method, params=None, session_id=None):
        if method == "initialize":
            return {"_session_id": "session-1"}
        assert method == "tools/list"
        assert session_id == "session-1"
        return {
            "tools": [
                {
                    "name": "allowed_tool",
                    "description": "Allowed",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "blocked_tool",
                    "description": "Blocked",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.mcp._json_rpc_call",
        mock_json_rpc_call,
    )

    tools = await discover_mcp_tools(
        hass,
        """
servers:
  - name: docs
    url: https://example.com/mcp
    allow_tools:
      - allowed_tool
""",
    )

    assert len(tools) == 1
    assert tools[0]["spec"]["name"] == "mcp__docs__allowed_tool"
    assert tools[0]["mcp"]["tool_name"] == "allowed_tool"


@pytest.mark.asyncio
async def test_discover_mcp_tools_skips_failing_server(
    hass: MagicMock, monkeypatch
) -> None:
    """Test failed MCP server is skipped while others still load."""

    async def mock_json_rpc_call(session, server, method, params=None, session_id=None):
        if method == "initialize":
            if server.name == "bad":
                raise McpServerError("failed")
            return {"_session_id": "session-good"}
        if server.name == "bad":
            raise McpServerError("failed")
        assert session_id == "session-good"
        return {
            "tools": [
                {
                    "name": "good_tool",
                    "description": "Good",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.mcp._json_rpc_call",
        mock_json_rpc_call,
    )

    tools = await discover_mcp_tools(
        hass,
        """
servers:
  - name: bad
    url: https://bad.example.com/mcp
  - name: good
    url: https://good.example.com/mcp
""",
    )

    assert len(tools) == 1
    assert tools[0]["spec"]["name"] == "mcp__good__good_tool"


@pytest.mark.asyncio
async def test_execute_mcp_tool(hass: MagicMock, monkeypatch) -> None:
    """Test executing an MCP tool call."""

    async def mock_json_rpc_call(session, server, method, params=None, session_id=None):
        if method == "initialize":
            return {"_session_id": "session-2"}
        assert method == "tools/call"
        assert session_id == "session-2"
        assert params == {"name": "search_docs", "arguments": {"query": "hello"}}
        return {"content": [{"type": "text", "text": "result text"}]}

    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.mcp._json_rpc_call",
        mock_json_rpc_call,
    )

    result = await execute_mcp_tool(
        hass,
        {
            "source": "mcp",
            "mcp": {
                "server_name": "docs",
                "url": "https://example.com/mcp",
                "transport": "streamable_http",
                "timeout": 20,
                "headers": {},
                "tool_name": "search_docs",
            },
        },
        {"query": "hello"},
    )

    assert result == "result text"
