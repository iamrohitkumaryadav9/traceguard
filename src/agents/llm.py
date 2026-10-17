"""LLM interface used by the agent nodes, and its Gemini implementation.

Nodes only see: llm.start_chat(system, tool_specs, purpose) -> chat.send(text | [ToolResult]) -> LLMTurn, and
llm.structured(system, prompt, schema, purpose) -> BaseModel. Tests supply a scripted FakeLLM
(tests/fakes.py) with the same two methods, so no test ever touches the network.
"""
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel

from src.agents import config


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ToolResult:
    name: str
    output: dict


@dataclass
class LLMTurn:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)  # prompt_tokens / output_tokens when the API reports them
    latency_ms: float = 0.0


def tool_specs_for(tools, hidden_args=config.HIDDEN_TOOL_ARGS):
    """[{name, description, parameters (JSON schema)}] from src.tools.TOOLS specs, minus hidden arguments."""
    specs = []
    for spec in tools:
        schema = spec.input_model.model_json_schema()
        props = {k: v for k, v in schema.get("properties", {}).items() if k not in hidden_args}
        for v in props.values():
            v.pop("title", None)
        specs.append({"name": spec.name, "description": spec.description,
                      "parameters": {"type": "object", "properties": props,
                                     "required": [r for r in schema.get("required", []) if r in props]}})
    return specs


class DailyQuotaExceeded(RuntimeError):
    """The API reports the DAILY quota is used up: waiting a minute will not help, so it is not retried."""


class GeminiLLM:
    """Gemini through google-genai: temperature 0, manual tool loop, throttled to the free-tier rate limit."""

    def __init__(self, model=None, client=None, min_interval_s=config.MIN_CALL_INTERVAL_S):
        self.model = model or config.LLM_MODEL
        self._client = client
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.throttle_wait_s = 0.0

    def _get_client(self):
        if self._client is None:
            from dotenv import load_dotenv
            from google import genai
            from google.genai import types

            load_dotenv()
            key = os.environ.get("GEMINI_API_KEY")
            if not key or key == "your_key_here":
                raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
            self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.HTTP_TIMEOUT_MS))
        return self._client

    def _generate(self, contents, config_obj):
        last = None
        for attempt in range(5):
            wait = max(0.0, self._last_call + self.min_interval_s - time.monotonic())
            if wait:
                time.sleep(wait)
                self.throttle_wait_s += wait
            try:
                t0 = time.perf_counter()
                resp = self._get_client().models.generate_content(model=self.model, contents=contents, config=config_obj)
                self._last_call = time.monotonic()
                return resp, 1000 * (time.perf_counter() - t0)
            except Exception as e:
                self._last_call = time.monotonic()
                last = e
                msg = str(e)
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and re.search(r"per ?day|daily", msg, re.I):
                    raise DailyQuotaExceeded(msg[:200]) from e
                pause =config.RATE_LIMIT_WAIT_S if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) else 2 ** attempt
                time.sleep(pause)
                self.throttle_wait_s += pause
        raise RuntimeError(f"Gemini call failed after retries: {type(last).__name__}")

    @staticmethod
    def _usage(resp):
        u = getattr(resp, "usage_metadata", None)
        if not u:
            return {}
        return {"prompt_tokens": getattr(u, "prompt_token_count", None), "output_tokens": getattr(u, "candidates_token_count", None)}

    def start_chat(self, system, tool_specs, purpose=""):
        return _GeminiChat(self, system, tool_specs)

    def structured(self, system, prompt, schema, purpose=""):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE, system_instruction=system,
            response_mime_type="application/json", response_schema=schema)
        resp, ms = self._generate([types.Content(role="user", parts=[types.Part.from_text(text=prompt)])], cfg)
        parsed = getattr(resp, "parsed", None)
        out = parsed if isinstance(parsed, schema) else schema.model_validate_json(resp.text)
        return out, {"usage": self._usage(resp), "latency_ms": ms}


class _GeminiChat:
    def __init__(self, llm: GeminiLLM, system, tool_specs):
        from google.genai import types

        self.llm, self.types = llm, types
        decls = [types.FunctionDeclaration(name=s["name"], description=s["description"],
                                           parameters_json_schema=s["parameters"]) for s in tool_specs]
        self.cfg = types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE, system_instruction=system,
            tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        self.contents = []

    def send(self, content) -> LLMTurn:
        t = self.types
        if isinstance(content, str):
            parts = [t.Part.from_text(text=content)]
        else:
            parts = [t.Part.from_function_response(name=r.name, response={"result": r.output}) for r in content]
        self.contents.append(t.Content(role="user", parts=parts))
        resp, ms = self.llm._generate(self.contents, self.cfg)
        cand = resp.candidates[0].content  # kept as is in the history (it carries Gemini's thought signatures)
        self.contents.append(cand)
        text, calls = [], []
        for p in cand.parts or []:
            if getattr(p, "function_call", None):
                calls.append(ToolCall(p.function_call.name, dict(p.function_call.args or {})))
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text.append(p.text)
        return LLMTurn(text="".join(text), tool_calls=calls, usage=self.llm._usage(resp), latency_ms=ms)
