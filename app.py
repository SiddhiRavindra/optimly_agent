"""Streamlit demo: ask a brand question, see the answer + evidence, and the eval results.

    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from optimly_agent.agent import AGENT_MODEL, BaselineAgent, OptimlyAgent

st.set_page_config(page_title="Optimly Brand Agent", page_icon="🔎", layout="wide")
st.title("Optimly Brand Agent")
st.caption(f"Answers grounded in the public Optimly Index · model: `{AGENT_MODEL}`")

RESULTS = Path(__file__).parent / "results"


@st.cache_resource
def agents():
    return OptimlyAgent(), BaselineAgent()


def show(result, title):
    st.subheader(title)
    if result.error and not result.answer:
        st.error(result.error)
        return
    a = result.answer
    if a.abstained:
        st.warning(f"**I don't know.** {a.reason or ''}")
        if a.answer:
            st.write(a.answer)
    else:
        if a.brand:
            st.markdown(f"**Brand:** {a.brand}")
        st.write(a.answer)
    if a.evidence:
        st.markdown("**Evidence**")
        for e in a.evidence:
            src = f"[{e.source}]({e.source})" if e.source.startswith("http") else e.source
            st.markdown(f"- {e.fact} — {src}")
    cost = f" · ${result.cost_usd:.5f}" if result.cost_usd is not None else ""
    tools = f" · tools: {', '.join(result.tool_calls)}" if result.tool_calls else ""
    st.caption(f"{result.latency_s:.1f}s · {result.input_tokens + result.output_tokens} tokens{cost}{tools}")


ask_tab, results_tab = st.tabs(["Ask", "Results"])

with ask_tab:
    q = st.text_input("Ask about a brand", placeholder="What category is CeraVe in?")
    compare = st.checkbox("Compare with baseline (no tools)")
    if st.button("Ask", type="primary") and q:
        optimly, baseline = agents()
        if compare:
            left, right = st.columns(2)
            with left, st.spinner("Optimly agent..."):
                show(optimly.ask(q), "Optimly agent")
            with right, st.spinner("Baseline..."):
                show(baseline.ask(q), "Baseline (no tools)")
        else:
            with st.spinner("Looking it up..."):
                show(optimly.ask(q), "Answer")

with results_tab:
    if (RESULTS / "metrics.csv").exists():
        st.subheader("Baseline vs Optimly agent")
        st.dataframe(pd.read_csv(RESULTS / "metrics.csv", index_col=0), use_container_width=True)
        judged = pd.read_csv(RESULTS / "judged.csv")
        st.subheader("Every answer")
        f = st.multiselect("Filter labels", sorted(judged["label"].unique()))
        st.dataframe(judged[judged["label"].isin(f)] if f else judged, use_container_width=True)
    else:
        st.info("No results yet. Run `python evaluate.py` first.")
