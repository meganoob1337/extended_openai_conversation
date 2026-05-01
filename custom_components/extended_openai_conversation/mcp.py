"""MCP tool discovery and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any
from uuid import uuid4

import aiohttp
import yaml

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .exceptions import McpConfigError, McpServerError, McpToolExecutionError

_LOGGER = logging.getLogger(__name__)

_VALID_TRANSPORTS = {"streamable_http", "http", "sse"}


@dataclass(slots=True)
class McpServerConfig:
    """Parsed MCP server configuration."""

    name: str
    url: str
    transport: str
    timeout: int
    headers: dict[str, str]
    allow_tools: list[str]


def _normalize_external_tool_name(server_name: str, tool_name: str) -> str:
    """Prefix MCP tool names to avoid collisions with local tools."""
    return f"mcp__{server_name}__{tool_name}"


def _format_mcp_result(result: Any) -> str:
    """Convert MCP tool result payload into text."""
    if isinstance(result, dict) and "content" in result and isinstance(
        result["content"], list
    ):
        parts: list[str] = []
        for item in result["content"]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            if item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=True))
        return "\n".join(part for part in parts if part)

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=True)


def parse_mcp_servers_config(raw_config: str | None) -> list[McpServerConfig]:
    """Parse MCP server config from YAML/JSON string."""
    if not raw_config or not raw_config.strip():
        return []

    try:
        parsed = yaml.safe_load(raw_config) or {}
    except yaml.YAMLError as err:
        raise McpConfigError("Invalid MCP YAML/JSON") from err

    if not isinstance(parsed, dict):
        raise McpConfigError("MCP configuration must be a mapping")

    servers = parsed.get("servers", [])
    if not isinstance(servers, list):
        raise McpConfigError("'servers' must be a list")

    result: list[McpServerConfig] = []
    for index, server in enumerate(servers):
        if not isinstance(server, dict):
            raise McpConfigError(f"Server at index {index} must be a mapping")

        name = str(server.get("name", "")).strip()
        url = str(server.get("url", "")).strip()
        transport = str(server.get("transport", "streamable_http")).strip().lower()
        timeout = int(server.get("timeout", 20))
        headers = server.get("headers", {}) or {}
        allow_tools = server.get("allow_tools", []) or []

        if not name:
            raise McpConfigError(f"Server at index {index} is missing 'name'")
        if not url:
            raise McpConfigError(f"Server '{name}' is missing 'url'")
        if transport not in _VALID_TRANSPORTS:
            raise McpConfigError(
                f"Server '{name}' has invalid transport '{transport}'"
            )
        if not isinstance(headers, dict):
            raise McpConfigError(f"Server '{name}' headers must be a mapping")
        if not isinstance(allow_tools, list):
            raise McpConfigError(f"Server '{name}' allow_tools must be a list")

        result.append(
            McpServerConfig(
                name=name,
                url=url,
                transport=transport,
                timeout=timeout,
                headers={str(k): str(v) for k, v in headers.items()},
                allow_tools=[str(tool_name) for tool_name in allow_tools],
            )
        )

    return result


async def _json_rpc_call(
    session: aiohttp.ClientSession,
    server: McpServerConfig,
    method: str,
    params: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Call JSON-RPC method on an MCP server."""
    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": method,
    }
    if params:
        payload["params"] = params

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **server.headers,
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    timeout = aiohttp.ClientTimeout(total=server.timeout)

    try:
        async with session.post(
            server.url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            response_session_id = response.headers.get("mcp-session-id")
    except Exception as err:
        raise McpServerError(
            f"Failed MCP call {method} on server '{server.name}': {err}"
        ) from err

    if not isinstance(data, dict):
        raise McpServerError(f"Invalid MCP response from server '{server.name}'")

    if "error" in data and data["error"]:
        raise McpServerError(
            f"MCP server '{server.name}' returned error for {method}: {data['error']}"
        )

    result = data.get("result")
    if not isinstance(result, dict):
        raise McpServerError(
            f"MCP server '{server.name}' returned invalid result for {method}"
        )
    if response_session_id:
        result = {**result, "_session_id": response_session_id}
    return result


async def _initialize_mcp_session(
    session: aiohttp.ClientSession,
    server: McpServerConfig,
) -> str:
    """Initialize MCP session and return session id."""
    init_result = await _json_rpc_call(
        session,
        server,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "homeassistant-extended-openai-conversation-test",
                "version": "1.0",
            },
        },
    )
    session_id = init_result.get("_session_id")
    if not isinstance(session_id, str) or not session_id:
        raise McpServerError(
            f"MCP server '{server.name}' did not return mcp-session-id during initialize"
        )
    return session_id


async def discover_mcp_tools(
    hass: HomeAssistant,
    raw_config: str | None,
) -> list[dict[str, Any]]:
    """Discover MCP tools and map to local function tool format."""
    servers = parse_mcp_servers_config(raw_config)
    if not servers:
        return []

    session = async_get_clientsession(hass)
    function_tools: list[dict[str, Any]] = []

    for server in servers:
        try:
            session_id = await _initialize_mcp_session(session, server)
            list_result = await _json_rpc_call(
                session, server, "tools/list", session_id=session_id
            )
        except McpServerError:
            _LOGGER.warning("Skipping MCP server '%s' due to connection error", server.name)
            continue

        tools = list_result.get("tools", [])
        if not isinstance(tools, list):
            _LOGGER.warning("Skipping MCP server '%s': invalid tools list", server.name)
            continue

        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name", "")).strip()
            if not tool_name:
                continue
            if server.allow_tools and tool_name not in server.allow_tools:
                continue

            description = str(tool.get("description", "")).strip()
            input_schema = tool.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}}

            function_tools.append(
                {
                    "spec": {
                        "name": _normalize_external_tool_name(server.name, tool_name),
                        "description": description or f"MCP tool {tool_name}",
                        "parameters": input_schema,
                    },
                    "source": "mcp",
                    "mcp": {
                        "server_name": server.name,
                        "url": server.url,
                        "transport": server.transport,
                        "timeout": server.timeout,
                        "headers": server.headers,
                        "tool_name": tool_name,
                    },
                }
            )

    return function_tools


async def execute_mcp_tool(
    hass: HomeAssistant,
    function_tool: dict[str, Any],
    arguments: dict[str, Any],
) -> str:
    """Execute MCP tool call and return a text result."""
    mcp_config = function_tool.get("mcp")
    if not isinstance(mcp_config, dict):
        raise McpToolExecutionError("Missing MCP tool metadata")

    server = McpServerConfig(
        name=str(mcp_config.get("server_name", "")),
        url=str(mcp_config.get("url", "")),
        transport=str(mcp_config.get("transport", "streamable_http")),
        timeout=int(mcp_config.get("timeout", 20)),
        headers=dict(mcp_config.get("headers", {})),
        allow_tools=[],
    )
    tool_name = str(mcp_config.get("tool_name", ""))
    if not tool_name:
        raise McpToolExecutionError("Missing MCP tool name")

    session = async_get_clientsession(hass)
    try:
        session_id = await _initialize_mcp_session(session, server)
        result = await _json_rpc_call(
            session,
            server,
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            session_id=session_id,
        )
    except McpServerError as err:
        raise McpToolExecutionError(str(err)) from err

    if result.get("isError"):
        raise McpToolExecutionError(
            f"MCP tool '{tool_name}' on server '{server.name}' returned an error"
        )

    return _format_mcp_result(result)
