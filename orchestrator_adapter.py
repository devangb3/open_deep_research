from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn


def _load_local_env() -> None:
    adapter_dir = Path(__file__).resolve().parent
    load_dotenv(adapter_dir / ".env", override=False)
    load_dotenv(override=False)


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


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _parse_csv(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or not raw.strip():
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or None


def _load_optional_text_file(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return text or None


def _load_final_report_system_prompt() -> Optional[str]:
    final_report_system_prompt_path = (
        Path(__file__).resolve().parents[1]
        / "whitepaper"
        / "browsecomp"
        / "prompts"
        / "odr_report_gen_system.txt"
    )
    return _load_optional_text_file(final_report_system_prompt_path)


class OpenDeepResearchRunner:
    """Lazy ODR graph wrapper for /responses execution."""

    def __init__(self) -> None:
        _load_local_env()
        self._graph = None

    def _configurable(self) -> Dict[str, Any]:
        configurable: Dict[str, Any] = {
            "allow_clarification": _parse_bool(os.getenv("ODR_ALLOW_CLARIFICATION"), False),
            "max_structured_output_retries": _parse_int(os.getenv("ODR_MAX_STRUCTURED_OUTPUT_RETRIES"), 3),
            "max_concurrent_research_units": _parse_int(os.getenv("ODR_MAX_CONCURRENT_RESEARCH_UNITS"), 4),
            "search_api": os.getenv("ODR_SEARCH_API", "tavily"),
            "openrouter_web_search_engine": os.getenv("ODR_OPENROUTER_WEB_SEARCH_ENGINE", "auto"),
            "openrouter_web_search_max_results": _parse_optional_int(
                os.getenv("ODR_OPENROUTER_WEB_SEARCH_MAX_RESULTS")
            ),
            "openrouter_web_search_max_total_results": _parse_optional_int(
                os.getenv("ODR_OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS")
            ),
            "openrouter_web_search_context_size": os.getenv("ODR_OPENROUTER_WEB_SEARCH_CONTEXT_SIZE"),
            "openrouter_web_search_allowed_domains": _parse_csv(
                os.getenv("ODR_OPENROUTER_WEB_SEARCH_ALLOWED_DOMAINS")
            ),
            "openrouter_web_search_excluded_domains": _parse_csv(
                os.getenv("ODR_OPENROUTER_WEB_SEARCH_EXCLUDED_DOMAINS")
            ),
            "max_researcher_iterations": _parse_int(os.getenv("ODR_MAX_RESEARCHER_ITERATIONS"), 4),
            "max_react_tool_calls": _parse_int(os.getenv("ODR_MAX_REACT_TOOL_CALLS"), 8),
            "summarization_model": os.getenv("ODR_SUMMARIZATION_MODEL", "openai:google/gemini-3.1-flash-lite-preview"),
            "summarization_model_max_tokens": _parse_optional_int(os.getenv("ODR_SUMMARIZATION_MODEL_MAX_TOKENS")),
            "research_model": os.getenv("ODR_RESEARCH_MODEL", "openai:google/gemini-3.1-pro-preview"),
            "research_model_max_tokens": _parse_optional_int(os.getenv("ODR_RESEARCH_MODEL_MAX_TOKENS")),
            "compression_model": os.getenv("ODR_COMPRESSION_MODEL", "openai:google/gemini-3.1-flash-lite-preview"),
            "compression_model_max_tokens": _parse_optional_int(os.getenv("ODR_COMPRESSION_MODEL_MAX_TOKENS")),
            "final_report_model": os.getenv("ODR_FINAL_REPORT_MODEL", "openai:google/gemini-3.1-pro-preview"),
            "final_report_model_max_tokens": _parse_optional_int(os.getenv("ODR_FINAL_REPORT_MODEL_MAX_TOKENS")),
        }
        final_report_system_prompt = _load_final_report_system_prompt()
        if final_report_system_prompt:
            configurable["final_report_system_prompt"] = final_report_system_prompt

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

    def runtime_config(
        self,
        *,
        thread_id: Optional[str] = None,
        lead_researcher_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        configurable = {
            "thread_id": thread_id or str(uuid.uuid4()),
            **self._configurable(),
        }
        if isinstance(lead_researcher_prompt_override, str) and lead_researcher_prompt_override.strip():
            configurable["lead_researcher_prompt_override"] = lead_researcher_prompt_override.strip()

        return {
            "configurable": configurable,
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

    async def run(
        self,
        input_messages: List[Dict[str, str]],
        *,
        lead_researcher_prompt_override: Optional[str] = None,
    ) -> str:
        graph = self._load_graph()
        config = self.runtime_config(lead_researcher_prompt_override=lead_researcher_prompt_override)

        if not any(message.get("role") == "user" and message.get("content", "").strip() for message in input_messages):
            raise RuntimeError("Open Deep Research requires at least one non-empty user message")

        final_state = await graph.ainvoke(
            {"messages": input_messages},
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


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n".join(parts).strip()


def _normalize_input_messages(input_payload: Any) -> List[Dict[str, str]]:
    if not isinstance(input_payload, list):
        return []

    normalized_messages: List[Dict[str, str]] = []
    for message in input_payload:
        if not isinstance(message, dict):
            continue

        raw_role = str(message.get("role") or "").strip().lower()
        if raw_role not in {"system", "developer", "user", "assistant"}:
            continue

        text = _extract_message_text(message.get("content"))
        if not text:
            continue

        role = "system" if raw_role in {"system", "developer"} else raw_role
        if normalized_messages and normalized_messages[-1]["role"] == role == "system":
            normalized_messages[-1]["content"] = f'{normalized_messages[-1]["content"]}\n\n{text}'
            continue

        normalized_messages.append({"role": role, "content": text})

    return normalized_messages


def _split_lead_researcher_prompt_override(
    input_messages: List[Dict[str, str]],
) -> tuple[Optional[str], List[Dict[str, str]]]:
    override_parts: List[str] = []
    forwarded_messages: List[Dict[str, str]] = []

    for message in input_messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "system":
            if content:
                override_parts.append(content)
            continue
        forwarded_messages.append(message)

    override = "\n\n".join(override_parts).strip() or None
    return override, forwarded_messages


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


def _stringify_error_detail(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        if isinstance(detail.get("message"), str) and detail["message"].strip():
            return detail["message"].strip()
        if isinstance(detail.get("detail"), str) and detail["detail"].strip():
            return detail["detail"].strip()
    return str(detail)


def _build_openrouter_like_error(message: str, *, error_type: str = "adapter_error") -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
        }
    }


def create_app() -> FastAPI:
    odr_runner = OpenDeepResearchRunner()
    tool_gateway = ODRToolGateway(runner=odr_runner)

    app = FastAPI(title="Open Deep Research Orchestrator Adapter", version="1.1.0")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_build_openrouter_like_error(_stringify_error_detail(exc.detail), error_type="http_error"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_build_openrouter_like_error(str(exc), error_type="internal_error"),
        )

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
        input_messages = _normalize_input_messages(input_payload)
        if not input_messages:
            raise HTTPException(status_code=400, detail="Unable to extract user query from input messages")
        lead_researcher_prompt_override, forwarded_messages = _split_lead_researcher_prompt_override(input_messages)
        if not forwarded_messages:
            raise HTTPException(status_code=400, detail="Unable to extract non-system input messages")

        model = str(payload.get("model") or os.getenv("ODR_ADAPTER_MODEL_ID", "open_deep_research"))

        try:
            answer = await odr_runner.run(
                forwarded_messages,
                lead_researcher_prompt_override=lead_researcher_prompt_override,
            )
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
