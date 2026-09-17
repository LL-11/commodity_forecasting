from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from mcp.server.mcpserver import MCPServer

from commodity_ai.agent import MCPResponsesAgent


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                type="function_call",
                name="market_fact",
                arguments='{"as_of":"2026-01-01"}',
                call_id="call-1",
            )
            return SimpleNamespace(id="response-1", output=[tool_call], output_text="")
        return SimpleNamespace(
            id="response-2", output=[], output_text="Storage was 3,100 Bcf [Demo citation]."
        )


class AgentTests(unittest.TestCase):
    def test_agent_discovers_and_calls_mcp_tool(self) -> None:
        server = MCPServer("agent-test")

        @server.tool()
        def market_fact(as_of: str) -> dict[str, object]:
            return {"as_of": as_of, "storage_bcf": 3100, "citation": "Demo citation"}

        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        agent = MCPResponsesAgent(server, openai_client=client, model="test-model")
        result = asyncio.run(agent.answer("What is storage?", "2026-01-01T00:00:00+00:00"))

        self.assertEqual(result.model, "test-model")
        self.assertIn("3,100", result.answer)
        self.assertEqual(result.tool_calls[0]["tool"], "market_fact")
        second_input = responses.calls[1]["input"][0]
        self.assertEqual(second_input["type"], "function_call_output")
        self.assertIn("3100", second_input["output"])


if __name__ == "__main__":
    unittest.main()
