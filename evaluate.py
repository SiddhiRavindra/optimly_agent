"""Run baseline vs Optimly agent on questions.json, judge every answer, build metrics.

Usage:
    python evaluate.py                 # full run
    python evaluate.py --limit 5       # quick smoke test
    python evaluate.py --rejudge       # re-judge saved answers without re-asking

Outputs (in results/):
    answers.jsonl   raw answers + latency/tokens per (system, question)
    judged.csv      one row per answer with the judge's label
    metrics.csv / metrics.md   the Baseline vs Optimly table
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from optimly_agent.agent import BaselineAgent, OptimlyAgent
from optimly_agent.schemas import JudgeVerdict
from optimly_agent.tools import fetch_profile

load_dotenv()

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:gpt-4o")
WORKERS = int(os.getenv("EVAL_WORKERS", "4"))

JUDGE_PROMPT = """You grade answers from a brand Q&A assistant.

You get: the question, its type, the expected answer/behavior, the reference facts
from the brand's Optimly profile (ground truth), and the assistant's structured answer.

Labels:
- correct: answered, and the answer matches the expected answer (minor extra detail is fine
  if it is consistent with the reference facts). For false-premise questions, explicitly
  correcting the premise counts as correct. For ambiguous questions, picking the intended
  brand while clearly stating the assumption, or listing the options, counts as correct.
- wrong: answered, but contradicts the expected answer or picks the wrong brand silently,
  or goes along with a false premise.
- unsupported: gives specific facts (numbers, names, dates, prices) that are NOT in the
  reference facts, even if they might be true. This is the "made up / unverifiable" label.
- correct_abstain: abstained or said it doesn't know, and that was the right behavior
  (info not in the data, false premise, or genuine ambiguity it flagged).
- wrong_abstain: abstained even though the reference facts contain the answer.

Judge only against the reference facts, not your own knowledge."""


def reference_facts(slug_field: str) -> str:
    blocks = []
    for slug in slug_field.split("|"):
        p = fetch_profile(slug)
        if "error" in p:
            blocks.append(f"[{slug}] {p['error']}")
            continue
        lines = [f"[{p['name']} | {p['domain']} | {p['profile_url']} | categories: {', '.join(p['categories'])}]"]
        lines += [f"- {f['section']}: {f['content']}" for f in p["facts"]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def ask_all(questions: list[dict], out_path: Path) -> list[dict]:
    systems = {"baseline": BaselineAgent(), "optimly": OptimlyAgent()}
    jobs = [(name, q) for name in systems for q in questions]

    def run(job):
        name, q = job
        r = systems[name].ask(q["question"])
        print(f"  [{name}] {q['id']} {r.latency_s:5.1f}s {'ERR' if r.error else ''}")
        return {
            "system": name, "id": q["id"],
            "answer": r.answer.model_dump() if r.answer else None,
            "error": r.error, "latency_s": r.latency_s,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cost_usd": r.cost_usd, "tool_calls": r.tool_calls,
            "forced_final": r.forced_final,
        }

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(run, jobs))
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return rows


def judge_all(questions: list[dict], answers: list[dict]) -> pd.DataFrame:
    judge = init_chat_model(JUDGE_MODEL, temperature=0).with_structured_output(JudgeVerdict)
    qmap = {q["id"]: q for q in questions}
    refs = {q["brand_slug"]: reference_facts(q["brand_slug"]) for q in questions}

    def grade(a):
        q = qmap[a["id"]]
        if a["answer"] is None:
            v = JudgeVerdict(label="wrong", rationale=f"No answer produced: {a['error']}")
        else:
            msg = (
                f"Question: {q['question']}\nType: {q['type']}\n"
                f"Expected behavior: {q['expected_behavior']}\nExpected answer: {q['expected_answer']}\n\n"
                f"Reference facts:\n{refs[q['brand_slug']]}\n\n"
                f"Assistant answer (JSON):\n{json.dumps(a['answer'], ensure_ascii=False)}"
            )
            try:
                v = judge.invoke([("system", JUDGE_PROMPT), ("user", msg)])
            except Exception as e:
                v = JudgeVerdict(label="wrong", rationale=f"Judge error: {e!r}")
        ans = a["answer"] or {}
        return {
            "system": a["system"], "id": a["id"], "type": q["type"], "question": q["question"],
            "expected": q["expected_answer"], "answer": ans.get("answer"),
            "abstained": ans.get("abstained"),
            "evidence": "; ".join(e["source"] for e in ans.get("evidence", [])),
            "label": v.label, "rationale": v.rationale,
            "latency_s": a["latency_s"], "tokens": a["input_tokens"] + a["output_tokens"],
            "cost_usd": a["cost_usd"], "tool_calls": ",".join(a.get("tool_calls") or []),
            "forced_final": a.get("forced_final", False),
        }

    with ThreadPoolExecutor(WORKERS) as ex:
        return pd.DataFrame(list(ex.map(grade, answers)))


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    def rate(sub: pd.DataFrame, labels: set[str]) -> float:
        return round(sub["label"].isin(labels).mean() * 100, 1) if len(sub) else float("nan")

    rows = {}
    for system, g in df.groupby("system"):
        by = {t: g[g["type"] == t] for t in ("answerable", "not_in_data", "ambiguous", "false_premise")}
        cost = g["cost_usd"].dropna()
        rows[system] = {
            "Accuracy on answerable (%)": rate(by["answerable"], {"correct"}),
            "Made-up facts rate, all Qs (%)": rate(g, {"unsupported"}),
            "Wrong answers, all Qs (%)": rate(g, {"wrong"}),
            "Correct 'I don't know' (not in data, %)": rate(by["not_in_data"], {"correct_abstain"}),
            "Ambiguity handled (%)": rate(by["ambiguous"], {"correct", "correct_abstain"}),
            "False premise handled (%)": rate(by["false_premise"], {"correct", "correct_abstain"}),
            "Over-abstained on answerable (%)": rate(by["answerable"], {"wrong_abstain"}),
            "Avg latency (s)": round(g["latency_s"].mean(), 2),
            "Avg tokens / question": round(g["tokens"].mean()),
            "Avg cost / question (USD)": round(cost.mean(), 5) if len(cost) else None,
        }
    table = pd.DataFrame(rows)
    return table.rename(columns={"baseline": "Baseline (no tools)", "optimly": "Optimly agent"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.add_argument("--rejudge", action="store_true", help="reuse results/answers.jsonl")
    args = p.parse_args()

    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    RESULTS.mkdir(exist_ok=True)
    answers_path = RESULTS / "answers.jsonl"

    if args.rejudge and answers_path.exists():
        ids = {q["id"] for q in questions}
        answers = [json.loads(l) for l in answers_path.read_text(encoding="utf-8").splitlines() if l]
        answers = [a for a in answers if a["id"] in ids]
    else:
        print(f"Asking {len(questions)} questions x 2 systems ...")
        answers = ask_all(questions, answers_path)

    print(f"Judging with {JUDGE_MODEL} ...")
    df = judge_all(questions, answers)
    df.to_csv(RESULTS / "judged.csv", index=False)

    table = metrics(df)
    table.to_csv(RESULTS / "metrics.csv")
    (RESULTS / "metrics.md").write_text(table.to_markdown(), encoding="utf-8")
    pd.set_option("display.width", 140)
    print("\n", table, "\n")
    print("Label counts:\n", pd.crosstab(df["system"], df["label"]))


if __name__ == "__main__":
    main()
