"""Agent tools. TOOLS maps name -> ToolSpec(description, input model, callable)."""
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from src.tools import (
    get_regulation_chunk,
    health,
    neighbors,
    score,
    search_regulations,
    subgraph,
    typology,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type
    fn: Callable


TOOLS = {
    m.NAME: ToolSpec(m.NAME, m.DESCRIPTION, getattr(m, "".join(w.capitalize() for w in m.NAME.split("_")) + "Input"),
                     getattr(m, m.NAME))
    for m in (score, neighbors, subgraph, typology, health, search_regulations, get_regulation_chunk)
}


def run_tool(name, arguments):
    """Validate `arguments` (dict) and call the tool; returns a JSON-serializable dict."""
    spec = TOOLS[name]
    out = spec.fn(spec.input_model(**arguments))
    return out.model_dump(mode="json") if isinstance(out, BaseModel) else out
