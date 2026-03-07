from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn


def _parse_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


class OpenDeepResearchRunner:
    """Lazy ODR graph wrapper for /responses execution."""

    def __init__(self) -> None:
        self._graph = None

    def _configurable(self) -> Dict[str, Any]:
        configurable: Dict[str, Any] = {
            "allow_clarification": _parse_bool(os.getenv("ODR_ALLOW_CLARIFICATION"), False),
            "max_structured_output_retries": _parse_int(os.getenv("ODR_MAX_STRUCTURED_OUTPUT_RETRIES"), 3),
            "max_concurrent_research_units": _parse_int(os.getenv("ODR_MAX_CONCURRENT_RESEARCH_UNITS"), 4),
            "search_api": os.getenv("ODR_SEARCH_API", "tavily"),
            "max_researcher_iterations": _parse_int(os.getenv("ODR_MAX_RESEARCHER_ITERATIONS"), 4),
            "max_react_tool_calls": _parse_int(os.getenv("ODR_MAX_REACT_TOOL_CALLS"), 8),
            "summarization_model": os.getenv("ODR_SUMMARIZATION_MODEL", "openai:gpt-4.1-mini"),
            "summarization_model_max_tokens": _parse_int(os.getenv("ODR_SUMMARIZATION_MODEL_MAX_TOKENS"), 4096),
            "research_model": os.getenv("ODR_RESEARCH_MODEL", "openai:gpt-4.1"),
            "research_model_max_tokens": _parse_int(os.getenv("ODR_RESEARCH_MODEL_MAX_TOKENS"), 8000),
            "compression_model": os.getenv("ODR_COMPRESSION_MODEL", "openai:gpt-4.1"),
            "compression_model_max_tokens": _parse_int(os.getenv("ODR_COMPRESSION_MODEL_MAX_TOKENS"), 8000),
            "final_report_model": os.getenv("ODR_FINAL_REPORT_MODEL", "openai:gpt-4.1"),
            "final_report_model_max_tokens": _parse_int(os.getenv("ODR_FINAL_REPORT_MODEL_MAX_TOKENS"), 8000),
        }

        mcp_url = os.getenv("ODR_MCP_URL")
        mcp_tools_raw = os.getenv("ODR_MCP_TOOLS")
        if mcp_url and mcp_tools_raw:
            mcp_tools = [item.strip() for item in mcp_tools_raw.split(",") if item.strip()]
            configurable["mcp_config"] = {
                "url": mcp_url,
                "tools": mcp_tools,
                "auth_required": _parse_bool(os.getenv("ODR_MCP_AUTH_REQUIRED"), False),
            }

        mcp_prompt = os.getenv("ODR_MCP_PROMPT")
        if mcp_prompt:
            configurable["mcp_prompt"] = mcp_prompt

        return configurable

    def runtime_config(self, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4()),
                **self._configurable(),
            },
            "metadata": {
                "owner": os.getenv("ODR_OWNER", "odr_adapter"),
            },
        }

    def _load_graph(self):
        if self._graph is not None:
            return self._graph

        import sys

        project_root = Path(__file__).resolve().parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from open_deep_research.deep_researcher import deep_researcher_builder
        from langgraph.checkpoint.memory import MemorySaver

        self._graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
        return self._graph

    @staticmethod
    def _extract_text_from_state(final_state: Dict[str, Any]) -> str:
        report = final_state.get("final_report")
        if isinstance(report, str) and report.strip():
            return report.strip()

        messages = final_state.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(last, dict):
                raw = last.get("content")
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()

        return ""

    async def run(self, user_query: str) -> str:
        graph = self._load_graph()
        config = self.runtime_config()

        final_state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config,
        )
        answer = self._extract_text_from_state(final_state)
        if not answer:
            raise RuntimeError("Open Deep Research produced an empty final report")
        return answer


class ODRToolGateway:
    """Expose exactly the executable tools ODR has access to (from get_all_tools)."""

    def __init__(self, runner: OpenDeepResearchRunner) -> None:
        self.runner = runner
        self._get_all_tools_fn: Optional[Callable[[Dict[str, Any]], Awaitable[List[Any]]]] = None

    def _load_get_all_tools_fn(self) -> Callable[[Dict[str, Any]], Awaitable[List[Any]]]:
        if self._get_all_tools_fn is not None:
            return self._get_all_tools_fn

        import sys

        project_root = Path(__file__).resolve().parent
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        from open_deep_research.utils import get_all_tools

        self._get_all_tools_fn = get_all_tools
        return self._get_all_tools_fn

    @staticmethod
    def _is_executable_tool(tool_obj: Any) -> bool:
        return hasattr(tool_obj, "name") and (hasattr(tool_obj, "ainvoke") or hasattr(tool_obj, "invoke"))

    @staticmethod
    def _extract_schema(tool_obj: Any) -> Dict[str, Any]:
        schema: Dict[str, Any] = {}

        args_schema = getattr(tool_obj, "args_schema", None)
        if isinstance(args_schema, dict):
            schema = dict(args_schema)
        elif args_schema is not None and hasattr(args_schema, "model_json_schema"):
            schema = dict(args_schema.model_json_schema())

        if not schema and hasattr(tool_obj, "get_input_schema"):
            try:
                input_schema = tool_obj.get_input_schema()
                if hasattr(input_schema, "model_json_schema"):
                    schema = dict(input_schema.model_json_schema())
            except Exception:
                schema = {}

        if not schema:
            schema = {"type": "object", "properties": {}}

        if schema.get("type") != "object":
            schema["type"] = "object"
        if not isinstance(schema.get("properties"), dict):
            schema["properties"] = {}
        if not isinstance(schema.get("required"), list):
            schema["required"] = []
        if not isinstance(schema.get("additionalProperties"), bool):
            schema["additionalProperties"] = False

        return schema

    async def _fetch_tools(self) -> List[Any]:
        get_all_tools = self._load_get_all_tools_fn()
        config = self.runner.runtime_config()
        tools = await get_all_tools(config)
        return list(tools or [])

    async def list_tools(self) -> List[Dict[str, Any]]:
        tools = await self._fetch_tools()
        manifest: List[Dict[str, Any]] = []
        for tool_obj in tools:
            if not self._is_executable_tool(tool_obj):
                continue
            manifest.append(
                {
                    "name": str(getattr(tool_obj, "name", "")).strip(),
                    "description": str(getattr(tool_obj, "description", "") or ""),
                    "parameters": self._extract_schema(tool_obj),
                }
            )
        return [item for item in manifest if item["name"]]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        tools = await self._fetch_tools()
        by_name = {
            str(getattr(tool_obj, "name", "")).strip(): tool_obj
            for tool_obj in tools
            if self._is_executable_tool(tool_obj)
        }

        selected = by_name.get(tool_name)
        if selected is None:
            raise ValueError(f"Unsupported tool_name: {tool_name}")

        config = self.runner.runtime_config()
        if hasattr(selected, "ainvoke"):
            result = await selected.ainvoke(arguments, config=config)
        else:
            result = selected.invoke(arguments, config=config)

        return jsonable_encoder(result)


def _extract_user_query(input_payload: Any) -> str:
    if not isinstance(input_payload, list):
        return ""

    for message in reversed(input_payload):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text

        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)

    return ""


def _build_openrouter_like_response(answer_text: str, model: str) -> Dict[str, Any]:
    now = int(time.time())
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": now,
        "model": model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": answer_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "parallel_tool_calls": False,
    }


def create_app() -> FastAPI:
    odr_runner = OpenDeepResearchRunner()
    tool_gateway = ODRToolGateway(runner=odr_runner)

    app = FastAPI(title="Open Deep Research Orchestrator Adapter", version="1.1.0")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "search_api": odr_runner._configurable().get("search_api", "tavily"),
        }

    @app.get("/tools")
    async def get_tools() -> Dict[str, Any]:
        try:
            tools = await tool_gateway.list_tools()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Tool discovery failed: {exc}") from exc
        return {"tools": tools}

    @app.post("/tools")
    async def execute_tool(request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments") or {}
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise HTTPException(status_code=400, detail="tool_name is required")
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=400, detail="arguments must be an object")

        tool_name = tool_name.strip()
        try:
            result = await tool_gateway.execute_tool(tool_name=tool_name, arguments=arguments)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {exc}") from exc

        return JSONResponse({"result": result})

    @app.post("/responses")
    async def responses(request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON payload")

        input_payload = payload.get("input")
        user_query = _extract_user_query(input_payload)
        if not user_query:
            raise HTTPException(status_code=400, detail="Unable to extract user query from input messages")

        model = str(payload.get("model") or os.getenv("ODR_ADAPTER_MODEL_ID", "open_deep_research"))

        try:
            answer = await odr_runner.run(user_query)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Open Deep Research execution failed: {exc}") from exc

        return JSONResponse(_build_openrouter_like_response(answer, model=model))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Deep Research adapter for PilotCrew orchestrator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4581)
    args = parser.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
