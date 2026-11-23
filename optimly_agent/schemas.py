"""Pydantic models for agent answers and judge verdicts."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    fact: str = Field(description="The specific fact used, quoted or closely paraphrased.")
    source: str = Field(description="Where it came from, e.g. the Optimly profile URL.")


class BrandAnswer(BaseModel):
    brand: Optional[str] = Field(None, description="The brand the answer is about, if identified.")
    answer: str = Field(description="The answer to the user's question.")
    evidence: list[Evidence] = Field(default_factory=list,
                                     description="Facts supporting the answer. Empty if none.")
    abstained: bool = Field(description="True if you cannot answer from what you know/were given.")
    reason: Optional[str] = Field(None, description="Why you abstained, if you did.")


Label = Literal["correct", "wrong", "unsupported", "correct_abstain", "wrong_abstain"]


class JudgeVerdict(BaseModel):
    label: Label = Field(description=(
        "correct: answered and matches the expected answer/behavior. "
        "wrong: answered but contradicts the expected answer. "
        "unsupported: states specific facts that are not in the reference data (made up). "
        "correct_abstain: abstained (or flagged ambiguity / rejected a false premise) when that was the right call. "
        "wrong_abstain: abstained even though the reference data contains the answer."
    ))
    rationale: str = Field(description="One or two sentences explaining the label.")
