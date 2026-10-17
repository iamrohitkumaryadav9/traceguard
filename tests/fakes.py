"""A scripted fake LLM with the same two methods as src.agents.llm.GeminiLLM. No network, ever."""
from src.agents.llm import LLMTurn, ToolCall
from src.agents.nodes.critic import JudgeOutput, JudgeVerdict


class FakeChat:
    def __init__(self, llm, purpose):
        self.llm, self.purpose = llm, purpose
        self.n = 0

    def send(self, content):
        self.llm.calls.append({"kind": "chat", "purpose": self.purpose, "content": content})
        script = self.llm.chat_scripts.get(self.purpose, [])
        if callable(script):
            turn = script(self.n, content)
        elif self.n < len(script):
            turn = script[self.n]
        else:
            turn = LLMTurn(text="DONE")
        self.n += 1
        return turn


class FakeLLM:
    throttle_wait_s = 0.0

    def __init__(self, chat_scripts=None, structured_scripts=None):
        self.chat_scripts = chat_scripts or {}
        self.structured_scripts = {k: list(v) if isinstance(v, list) else v for k, v in (structured_scripts or {}).items()}
        self.calls = []
        self.chats_started = []

    def start_chat(self, system, tool_specs, purpose=""):
        self.chats_started.append({"purpose": purpose, "tools": [s["name"] for s in tool_specs], "specs": tool_specs})
        return FakeChat(self, purpose)

    def structured(self, system, prompt, schema, purpose=""):
        self.calls.append({"kind": "structured", "purpose": purpose, "prompt": prompt})
        script = self.structured_scripts.get(purpose)
        if callable(script):
            out = script(prompt)
        elif script:
            out = script.pop(0) if len(script) > 1 else script[0]  # the last item repeats
            if callable(out):
                out = out(prompt)
        else:
            raise AssertionError(f"no structured script for {purpose!r}")
        return out, {"usage": {}, "latency_ms": 0.0}

    def count(self, kind=None, purpose=None):
        return sum(1 for c in self.calls if (kind is None or c["kind"] == kind) and (purpose is None or c["purpose"] == purpose))


def call(name, **args):
    return LLMTurn(tool_calls=[ToolCall(name, args)])


def done():
    return LLMTurn(text="DONE")


def all_supported(prompt):
    n = prompt.count("ITEM ")
    return JudgeOutput(verdicts=[JudgeVerdict(index=i, verdict="supported", reason="ok") for i in range(n)])
