"""Tests for MCP execution dispatch in entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.extended_openai_conversation.entity import ExtendedOpenAIBaseLLMEntity


@pytest.mark.asyncio
async def test_execute_function_tool_routes_to_mcp(hass: MagicMock, monkeypatch) -> None:
    """Ensure MCP tools are executed via MCP runtime helper."""
    entry = MagicMock()
    entry.runtime_data = MagicMock()

    subentry = MagicMock()
    subentry.subentry_id = "sub-1"
    subentry.title = "Assistant"
    subentry.data = {}

    entity = ExtendedOpenAIBaseLLMEntity(entry, subentry)
    entity.hass = hass
    entity.entity_id = "conversation.test"

    mock_execute = AsyncMock(return_value="mcp result")
    monkeypatch.setattr(
        "custom_components.extended_openai_conversation.entity.execute_mcp_tool",
        mock_execute,
    )

    tool_input = MagicMock()
    tool_input.id = "tool-1"
    tool_input.tool_name = "mcp__docs__search"
    tool_input.tool_args = {"query": "lights"}

    tool_result = await entity._execute_function_tool(
        {"source": "mcp", "mcp": {"server_name": "docs", "tool_name": "search"}},
        tool_input,
        llm_context=None,
        exposed_entities=[],
    )

    mock_execute.assert_awaited_once()
    assert tool_result.tool_name == "mcp__docs__search"
    assert tool_result.tool_result["result"] == "mcp result"
