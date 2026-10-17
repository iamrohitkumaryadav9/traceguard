"""CLI: python -m src.agents.run <tx_id>   (uses Gemini via GEMINI_API_KEY from .env)"""
import json
import sys

from src.agents import config
from src.agents.graph import run_agent, trace_dict
from src.agents.llm import GeminiLLM
from src.agents.memo import memo_to_markdown
from src.agents.runtime import Deps


def run_one(tx_id: int, llm=None, trace_dir=None):
    deps = Deps(llm=llm or GeminiLLM())
    state = run_agent(tx_id, deps)
    trace = trace_dict(state, deps)
    out_dir = trace_dir or config.TRACE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tx_id}.json").write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return state, trace


def main(argv):
    if len(argv) != 1:
        print("usage: python -m src.agents.run <tx_id>")
        return 2
    state, trace = run_one(int(argv[0]))
    print(memo_to_markdown(state.memo))
    print(f"\nstatus={state.status}; counts={trace['counts']}; latency_s={trace['latency_s']}; trace: {config.TRACE_DIR / (argv[0] + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
