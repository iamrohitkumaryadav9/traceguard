"""Agent configuration. One place for the model name and every budget."""
from pathlib import Path

LLM_MODEL = "gemini-3.5-flash-lite"  # the only place the model name is set
LLM_TEMPERATURE = 0

MAX_TOOL_CALLS = 8  # investigator tool calls per run
MAX_SEARCHES = 3  # search_regulations calls per run
MAX_CHUNK_FETCHES = 3  # get_regulation_chunk calls by the regulation node per run
MAX_CRITIC_LOOPS = 2  # rewrites after a failed critic round
MAX_LLM_CALLS = 30  # every LLM call in a run, all nodes

MIN_CALL_INTERVAL_S = 4.5  # the free tier allows 15 requests per minute per model
RATE_LIMIT_WAIT_S = 62

TRACE_DIR = Path("eval/traces")
INVESTIGATOR_TOOLS = ("get_neighbors", "subgraph_summary", "match_typology")
REGULATION_TOOLS = ("search_regulations", "get_regulation_chunk")
HIDDEN_TOOL_ARGS = ("rewrite", "rerank")  # retrieval is frozen: the LLM cannot switch these on
