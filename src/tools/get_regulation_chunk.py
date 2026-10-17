from typing import Union

from pydantic import BaseModel

from src.rag.retriever import get_retriever
from src.tools.schemas import ToolError

NAME = "get_regulation_chunk"
DESCRIPTION = (
    "Return the exact stored text of one regulation chunk by evidence_id (format reg:<doc_id>:p<page>:c<n>), "
    "so that a quotation can be verified character for character."
)


class GetRegulationChunkInput(BaseModel):
    evidence_id: str


class GetRegulationChunkOutput(BaseModel):
    evidence_id: str
    doc_id: str
    title: str
    publisher: str
    year: int
    jurisdiction: str
    page_start: int
    page_end: int
    text: str


def get_regulation_chunk(args: GetRegulationChunkInput) -> Union[GetRegulationChunkOutput, ToolError]:
    c = get_retriever().get(args.evidence_id)
    if c is None:
        return ToolError(error="unknown_evidence_id", tool=NAME, evidence_id=f"error:{NAME}:{args.evidence_id}",
                         message=f"no chunk with evidence_id {args.evidence_id!r}")
    return GetRegulationChunkOutput(
        evidence_id=c["evidence_id"], doc_id=c["doc_id"], title=c["title"], publisher=c["publisher"],
        year=c["year"], jurisdiction=c["jurisdiction"], page_start=c["page_start"], page_end=c["page_end"],
        text=c["text"])
