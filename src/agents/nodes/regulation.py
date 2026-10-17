"""Regulation: looks up text for patterns that have a retrieval_query (so never for lowers_risk patterns)."""
from src.agents import config, prompts
from src.agents.nodes.investigator import compact, run_tool_loop
from src.agents.runtime import Deps, call_tool
from src.agents.state import AgentState


def retrieval_queries(state: AgentState):
    """[(pattern, query)] for matched patterns whose retrieval_query is not null."""
    out = []
    for e in state.ledger.values():
        if e.tool == "match_typology" and "matches" in e.output:
            out += [(m["pattern"], m["retrieval_query"]) for m in e.output["matches"] if m.get("retrieval_query")]
    return out


def add_search_hits_to_ledger(state: AgentState):
    """Make every retrieved chunk citable under its own evidence_id (the search output holds the hits)."""
    from src.agents.state import LedgerEntry

    for eid, e in list(state.ledger.items()):
        if e.tool == "search_regulations" and "hits" in e.output:
            for h in e.output["hits"]:
                if h["evidence_id"] not in state.ledger:
                    state.ledger[h["evidence_id"]] = LedgerEntry(tool="search_regulations", args=e.args, output=h)


def regulation(state: AgentState, deps: Deps) -> AgentState:
    from src.tools import TOOLS

    state.nodes_visited.append("regulation")
    queries = retrieval_queries(state)
    first = ("Patterns matched for this transaction, with suggested retrieval queries:\n"
             + compact([{"pattern": p, "retrieval_query": q} for p, q in queries])
             + f"\nSearch at most {config.MAX_SEARCHES} times.")
    run_tool_loop(state, deps, "regulation", config.REGULATION_TOOLS,
                  prompts.REGULATION.format(max_searches=config.MAX_SEARCHES, max_fetches=config.MAX_CHUNK_FETCHES), first,
                  {"search_regulations": "search_count", "get_regulation_chunk": "chunk_fetch_count"}, TOOLS)
    add_search_hits_to_ledger(state)
    return state
