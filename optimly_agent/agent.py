"""Two answerers with the same model and output format:

  - baseline: the model alone, no tools (it may abstain)
  - optimly:  the model + Optimly Index tools (search_brand, get_brand_profile)

Both return a RunResult with the structured answer, latency, tokens and cost.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from .schemas import BrandAnswer
from .tools import TOOLS

load_dotenv()
# with_structured_output(include_raw=True) triggers a harmless serializer warning.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")

# USD per 1M tokens (input, output). Add your model here if it's missing.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}

OPTIMLY_PROMPT = """You answer questions about brands using the Optimly Index.

Rules:
1. Always call search_brand first, then get_brand_profile for the best match, before answering.
2. Use only facts returned by the tools. Cite the profile_url as the source of each fact.
3. If several different companies match the name (not just sub-brands of one company)
   and the question doesn't make clear which one, say the name is ambiguous, list the
   options, and set abstained=true with reason "ambiguous".
4. If the profile doesn't contain the answer, set abstained=true and explain what is missing.
   Don't go hunting through sub-brand profiles (e.g. "Notion AI", "figma-figjam") for it,
   and never fetch the same slug twice. Fetch at most 3 profiles per question.
5. If the question assumes something the profile doesn't support (a false premise),
   say so instead of going along with it.
6. Never guess or fill in from memory."""

STEP_BUDGET = 16  # graph steps (~7 tool calls) before we force a final answer

FINALIZE_PROMPT = """Stop searching. Answer the original question now using ONLY the tool
results above. If they don't contain the answer, set abstained=true and say what is missing.
If the question's premise isn't supported by them, say so. Cite profile URLs as sources."""

BASELINE_PROMPT ="""You answer questions about brands from your own knowledge.
Give a short, factual answer. If you are not confident, or the question can't be
answered reliably, set abstained=true and explain why. If the question assumes
something false, say so. Do not make up facts. Evidence may cite your general
knowledge as the source."""


@dataclass
class RunResult:
    answer: Optional[BrandAnswer]
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    tool_calls: list[str] = field(default_factory=list)
    error: Optional[str] = None
    forced_final: bool = False  # True if the step budget ran out and we forced an answer


def _cost(model: str, inp: int, out: int) -> Optional[float]:
    name = model.split(":", 1)[-1]
    if name not in PRICES:
        return None
    pi, po = PRICES[name]
    return (inp * pi + out * po) / 1_000_000


def _usage(messages: list[Any]) -> tuple[int, int]:
    inp = out = 0
    for m in messages:
        u = getattr(m, "usage_metadata", None)
        if u:
            inp += u.get("input_tokens", 0)
            out += u.get("output_tokens", 0)
    return inp, out


class OptimlyAgent:
    def __init__(self, model: str = AGENT_MODEL):
        self.model = model
        self.graph = create_agent(
            model=init_chat_model(model, temperature=0),
            tools=TOOLS,
            system_prompt=OPTIMLY_PROMPT,
            response_format=BrandAnswer,
            # Hard cap: once exceeded, further calls are blocked and the model must answer.
            middleware=[
                ToolCallLimitMiddleware(tool_name="get_brand_profile", run_limit=3),
                ToolCallLimitMiddleware(tool_name="search_brand", run_limit=3),
            ],
        )
        self.finalizer = init_chat_model(model, temperature=0).with_structured_output(
            BrandAnswer, include_raw=True)

    def ask(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        state: dict[str, Any] = {}
        try:
            # stream_mode="values" keeps the latest full state, so we still have the
            # gathered tool results if the agent loops past its step budget.
            for state in self.graph.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": STEP_BUDGET}, stream_mode="values",
            ):
                pass
        except GraphRecursionError:
            pass  # handled by the forced final answer below
        except Exception as e:  # keep evaluation running
            return RunResult(None, time.perf_counter() - t0, error=repr(e))

        msgs = list(state.get("messages", []))
        answer = state.get("structured_response")
        forced = False
        if answer is None and msgs:
            # Agent kept searching instead of answering: answer from what it already found.
            raw = self.finalizer.invoke(msgs + [HumanMessage(FINALIZE_PROMPT)])
            answer, forced = raw["parsed"], True
            msgs.append(raw["raw"])
        latency = time.perf_counter() - t0
        inp, outp = _usage(msgs)
        calls = [tc["name"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls]
        return RunResult(answer, latency, inp, outp, _cost(self.model, inp, outp), calls,
                         error=None if answer else "no answer", forced_final=forced)


class BaselineAgent:
    def __init__(self, model: str = AGENT_MODEL):
        self.model = model
        llm = init_chat_model(model, temperature=0)
        self.chain = llm.with_structured_output(BrandAnswer, include_raw=True)

    def ask(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        try:
            out = self.chain.invoke([("system", BASELINE_PROMPT), ("user", question)])
        except Exception as e:
            return RunResult(None, time.perf_counter() - t0, error=repr(e))
        latency = time.perf_counter() - t0
        inp, outp = _usage([out["raw"]])
        return RunResult(out.get("parsed"), latency, inp, outp, _cost(self.model, inp, outp),
                         error=repr(out["parsing_error"]) if out.get("parsing_error") else None)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What category is CeraVe in?"
    for name, agent in (("baseline", BaselineAgent()), ("optimly", OptimlyAgent())):
        r = agent.ask(q)
        print(f"\n=== {name} ({r.latency_s:.1f}s, {r.input_tokens}+{r.output_tokens} tok, tools={r.tool_calls})")
        print(r.answer.model_dump_json(indent=2) if r.answer else r.error)
