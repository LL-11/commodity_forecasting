from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

try:
    from mcp import Client
except ImportError as error:  # pragma: no cover - optional adapter
    raise RuntimeError("Install the agent dependencies: pip install -e '.[agent,mcp]'") from error


AGENT_INSTRUCTIONS = """You are a Henry Hub natural-gas market analyst.
Use the MCP tools for every market fact, forecast value, driver, metric, or report claim.
Tool argument keys must exactly match the JSON schema's snake_case names; never title-case them
or replace underscores with spaces. The get_weather_signal horizon must be 7 or 14; use 14 when
analyzing a longer forecast horizon. For get_henry_hub_history, start_date and end_date must be
calendar dates in YYYY-MM-DD format, not timestamps. Start report retrieval without metadata
filters; add a filter only when its exact value came from an earlier tool result.
Never calculate, alter, round, or invent a numerical forecast: reproduce numerical model output exactly.
Clearly separate model-derived drivers from narrative report evidence. State the as-of timestamp.
When using report evidence, cite the returned citation and source URL. Never use evidence published
after the as-of timestamp. Describe P10/P90 as an empirically calibrated uncertainty interval, not a
guarantee. If tools do not provide enough evidence, say so. This is educational, not trading advice.
"""


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    model: str
    tool_calls: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {"answer": self.answer, "model": self.model, "tool_calls": list(self.tool_calls)}


class MCPResponsesAgent:
    """OpenAI Responses agent whose only market capabilities come from an MCP server."""

    def __init__(
        self,
        server: Any | None = None,
        *,
        openai_client: Any | None = None,
        model: str | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        if server is None:
            server = os.getenv("COMMODITY_AI_MCP_URL")
            if not server:
                from .mcp_server import mcp

                server = mcp
        if openai_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:  # pragma: no cover - optional adapter
                raise RuntimeError("Install the agent extra: pip install -e '.[agent]'") from error
            openai_client = AsyncOpenAI()
        self.server = server
        self.openai_client: Any = openai_client
        self.model: str = model or os.getenv("OPENAI_MODEL") or "gpt-5.6-sol"
        self.max_tool_rounds = max_tool_rounds

    @staticmethod
    def _response_tools(listing: Any) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or f"Call the {tool.name} MCP tool.",
                "parameters": tool.input_schema,
                "strict": False,
            }
            for tool in listing.tools
        ]

    async def answer(self, question: str, as_of: str) -> AgentAnswer:
        if not question.strip():
            raise ValueError("question cannot be empty")
        traces: list[dict[str, object]] = []
        async with Client(self.server) as mcp_client:
            tools = self._response_tools(await mcp_client.list_tools())
            responses_api: Any = self.openai_client.responses
            response: Any = await responses_api.create(
                model=self.model,
                instructions=AGENT_INSTRUCTIONS,
                input=(
                    f"As-of timestamp: {as_of}\n"
                    f"User question: {question}\n"
                    "Use the available MCP tools before answering."
                ),
                tools=tools,
            )
            for _ in range(self.max_tool_rounds):
                calls = [item for item in response.output if item.type == "function_call"]
                if not calls:
                    answer = response.output_text.strip()
                    if not answer:
                        raise RuntimeError("the language model returned no answer")
                    return AgentAnswer(answer, self.model, tuple(traces))
                outputs: list[dict[str, str]] = []
                for call in calls:
                    arguments = json.loads(call.arguments or "{}")
                    result = await mcp_client.call_tool(call.name, arguments)
                    serialized = result.model_dump(mode="json", by_alias=True)
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(serialized, default=str),
                        }
                    )
                    traces.append(
                        {
                            "tool": call.name,
                            "arguments": arguments,
                            "is_error": bool(result.is_error),
                        }
                    )
                response = await responses_api.create(
                    model=self.model,
                    instructions=AGENT_INSTRUCTIONS,
                    previous_response_id=response.id,
                    input=outputs,
                    tools=tools,
                )
        raise RuntimeError("the agent exceeded its MCP tool-call limit")

    def answer_sync(self, question: str, as_of: str) -> AgentAnswer:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.answer(question, as_of))
        raise RuntimeError("answer_sync cannot run inside an active event loop; await answer()")
