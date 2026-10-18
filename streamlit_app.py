"""Streamlit dashboard for the Campus Café CrewAI launch workflow."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import streamlit as st

from cafe_launch import ASSUMPTIONS, run_workflow


ROOT = Path(__file__).resolve().parent
BRIEF_PATH = ROOT / "final_brief.md"
REFLECTION_PATH = ROOT / "reflection.md"
URL_PATTERN = re.compile(r"https?://[^\s)\]>\"']+")


def task_output(result: Any, index: int) -> str:
    outputs = getattr(result, "tasks_output", [])
    if index >= len(outputs):
        return ""
    output = outputs[index]
    return str(getattr(output, "raw", output)).strip()


def financial_metrics() -> dict[str, float | int]:
    assumptions = ASSUMPTIONS
    fixed_cost = sum(
        assumptions[key]
        for key in (
            "monthly_space_cost_inr",
            "monthly_staff_cost_inr",
            "monthly_utilities_inr",
            "monthly_other_fixed_costs_inr",
        )
    )
    contribution = (
        assumptions["average_transaction_inr"]
        - assumptions["variable_cost_per_transaction_inr"]
    )
    monthly_break_even = math.ceil(fixed_cost / contribution)
    daily_break_even = math.ceil(
        monthly_break_even / assumptions["operating_days_per_month"]
    )
    target_monthly_orders = (
        assumptions["target_transactions_per_day"]
        * assumptions["operating_days_per_month"]
    )
    target_contribution = target_monthly_orders * contribution
    return {
        "fixed_cost": fixed_cost,
        "contribution": contribution,
        "monthly_break_even": monthly_break_even,
        "daily_break_even": daily_break_even,
        "enrollment_share": daily_break_even / assumptions["enrolled_students"],
        "target_monthly_orders": target_monthly_orders,
        "target_contribution": target_contribution,
        "target_surplus": target_contribution - fixed_cost,
    }


def collect_urls(*texts: str) -> list[str]:
    urls: list[str] = []
    for text in texts:
        for url in URL_PATTERN.findall(text):
            cleaned = url.rstrip(".,;:")
            if cleaned not in urls:
                urls.append(cleaned)
    return urls


def read_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_markdown_without_heading_anchors(markdown: str) -> None:
    """Render Markdown headings as bold text to avoid Streamlit anchor links."""
    markdown = re.sub(
        r"(?m)^#{1,6}\s+(.+?)\s*#*\s*$",
        r"**\1**",
        markdown,
    )
    st.markdown(markdown)


def safe_error(error: Exception) -> str:
    message = str(error)
    for variable in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        secret = os.getenv(variable)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message or "No additional error details were provided."


st.set_page_config(
    page_title="Campus Café Launch Dashboard",
    page_icon="☕",
    layout="wide",
)
st.markdown(
    """
    <style>
    .stApp { background: #f6f7f4; }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e6e9e2;
        padding: 1rem; border-radius: 0.8rem;
    }
    .hero {
        background: linear-gradient(115deg, #163b32, #28634d);
        color: white; padding: 1.6rem 2rem; border-radius: 1rem;
        margin-bottom: 1rem;
    }
    .hero p { color: #e5efe8; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Campus Café Launch Dashboard</h1>
      <p>Evidence, unit economics and a reviewed recommendation for an
      8-week campus café pilot.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "launch_result" not in st.session_state:
    st.session_state.launch_result = None
if "launch_error" not in st.session_state:
    st.session_state.launch_error = None

if st.button("Generate Launch Brief", type="primary", use_container_width=True):
    st.session_state.launch_error = None
    try:
        with st.spinner(
            "Agents are researching sources, checking the financial model, "
            "drafting and reviewing the brief..."
        ):
            st.session_state.launch_result = run_workflow(
                output_path=BRIEF_PATH
            )
    except Exception as error:
        st.session_state.launch_result = None
        st.session_state.launch_error = (
            f"{type(error).__name__}: {safe_error(error)}"
        )

if st.session_state.launch_error:
    st.error(
        "The brief was not generated. Check that GROQ_API_KEY and "
        "TAVILY_API_KEY are configured, then retry. "
        + st.session_state.launch_error
    )
    st.info(
        "The existing brief, if any, was left unchanged. Search/tool failures "
        "are surfaced rather than replaced with invented research."
    )

result = st.session_state.launch_result
research_text = task_output(result, 0) if result is not None else ""
finance_text = task_output(result, 1) if result is not None else ""
sustainability_text = task_output(result, 2) if result is not None else ""
review_text = task_output(result, 4) if result is not None else ""
brief_text = read_markdown(BRIEF_PATH)
reflection_text = read_markdown(REFLECTION_PATH)
metrics = financial_metrics()

recommendation = "Run a measured 8-week pilot; do not scale until local demand and costs are validated."
if brief_text:
    first_paragraphs = brief_text.split("\n\n")
    for paragraph in first_paragraphs:
        if "recommend" in paragraph.lower() or "pilot" in paragraph.lower():
            recommendation = re.sub(r"[*#]", "", paragraph).strip()
            break

tabs = st.tabs(
    [
        "Campus Café Dashboard",
        "Agent Workflow",
        "Market Research and Sources",
        "Financial Break-Even Analysis",
        "Critical Review",
        "Final Launch Brief",
        "Project Reflection",
    ]
)

with tabs[0]:
    st.subheader("Pilot recommendation")
    st.success(recommendation)
    st.caption(
        "Planning case only: local student demand, supplier quotes and "
        "competitor pricing have not been validated."
    )
    cols = st.columns(4)
    cols[0].metric("Fixed costs / month", f"₹{metrics['fixed_cost']:,.0f}")
    cols[1].metric("Contribution / order", f"₹{metrics['contribution']:,.0f}")
    cols[2].metric("Break-even / month", f"{metrics['monthly_break_even']:,} orders")
    cols[3].metric("Target-case surplus / month", f"₹{metrics['target_surplus']:,.0f}")
    st.markdown("### Decision at a glance")
    st.write(
        f"The illustrative model requires **{metrics['daily_break_even']} "
        f"orders per operating day** to cover monthly operating costs. The "
        f"target case assumes **{ASSUMPTIONS['target_transactions_per_day']} "
        "orders/day**. This is not a sales forecast."
    )
    if brief_text:
        st.download_button(
            "Download final_brief.md",
            data=brief_text,
            file_name="final_brief.md",
            mime="text/markdown",
            key="download_final_brief_dashboard",
        )

with tabs[1]:
    st.subheader("Agent Workflow")
    st.write(
        "The existing five-agent workflow is preserved. Task outputs are "
        "passed forward explicitly: research, finance and sustainability feed "
        "the writer; all four upstream outputs feed the skeptical reviewer."
    )
    agents = [
        ("Market Researcher", "Finds source-linked market evidence and states scope limits."),
        ("Financial Analyst", "Calculates break-even and target-case economics."),
        ("Marketing Writer", "Turns task evidence into a concise decision brief."),
        ("Skeptical Reviewer", "Checks claims, arithmetic, pricing assumptions and risks."),
        ("Sustainability Officer", "Checks sustainability evidence and measurable pilot actions."),
    ]
    for start in range(0, len(agents), 2):
        cols = st.columns(2)
        for col, (name, role) in zip(cols, agents[start : start + 2]):
            with col:
                st.markdown(f"**{name}**")
                st.write(role)
    st.caption(
        "All agents and tasks are created by cafe_launch.py and run in the "
        "existing sequential CrewAI process."
    )

with tabs[2]:
    st.subheader("Market Research and Sources")
    st.warning(
        "Research findings are shown only after a successful workflow run. "
        "No local campus survey or verified campus competitor dataset is included."
    )
    if research_text:
        st.markdown("#### Market Researcher output")
        st.write(research_text)
    if sustainability_text:
        st.markdown("#### Sustainability Officer output")
        st.write(sustainability_text)
    sources = collect_urls(research_text, sustainability_text, brief_text)
    st.markdown("#### Clickable source links")
    if sources:
        for source in sources:
            st.markdown(f"- [{source}]({source})")
    else:
        st.info(
            "No research URLs are available yet. Generate the brief to run "
            "web search; if search fails, the application will report the error."
        )
    st.caption(
        "Sources shown here come from agent outputs or the saved brief; "
        "they are not independently endorsed by the dashboard."
    )

with tabs[3]:
    st.subheader("Financial Break-Even Analysis")
    st.markdown("#### Planning assumptions")
    left, right = st.columns(2)
    with left:
        st.write(f"Average order value: **₹{ASSUMPTIONS['average_transaction_inr']:,.0f}**")
        st.write(
            "Variable cost per order: "
            f"**₹{ASSUMPTIONS['variable_cost_per_transaction_inr']:,.0f}**"
        )
        st.write(f"Operating days/month: **{ASSUMPTIONS['operating_days_per_month']}**")
        st.write(
            f"Enrollment denominator: **{ASSUMPTIONS['enrolled_students']:,} students** "
            "(scenario input, not measured daily footfall)"
        )
    with right:
        for label, key in (
            ("Space", "monthly_space_cost_inr"),
            ("Staff", "monthly_staff_cost_inr"),
            ("Utilities", "monthly_utilities_inr"),
            ("Other fixed costs", "monthly_other_fixed_costs_inr"),
        ):
            st.write(f"{label}: **₹{ASSUMPTIONS[key]:,.0f}/month**")
        st.write(f"Target volume: **{ASSUMPTIONS['target_transactions_per_day']} orders/day**")
    st.markdown("#### Calculations")
    st.code(
        f"Monthly fixed costs = ₹{metrics['fixed_cost']:,.0f}\n"
        f"Contribution/order = ₹{ASSUMPTIONS['average_transaction_inr']} "
        f"− ₹{ASSUMPTIONS['variable_cost_per_transaction_inr']} "
        f"= ₹{metrics['contribution']:,.0f}\n"
        f"Monthly break-even = ceil(₹{metrics['fixed_cost']:,.0f} / "
        f"₹{metrics['contribution']:,.0f}) = {metrics['monthly_break_even']:,} orders\n"
        f"Daily break-even = ceil({metrics['monthly_break_even']:,} / "
        f"{ASSUMPTIONS['operating_days_per_month']}) = {metrics['daily_break_even']} orders/day\n"
        f"Enrollment share = {metrics['daily_break_even']} / "
        f"{ASSUMPTIONS['enrolled_students']:,} = {metrics['enrollment_share']:.2%}\n"
        f"Target monthly contribution = {ASSUMPTIONS['target_transactions_per_day']} "
        f"× {ASSUMPTIONS['operating_days_per_month']} × ₹{metrics['contribution']:,.0f} "
        f"= ₹{metrics['target_contribution']:,.0f}\n"
        f"Target-case operating surplus = ₹{metrics['target_contribution']:,.0f} "
        f"− ₹{metrics['fixed_cost']:,.0f} = ₹{metrics['target_surplus']:,.0f}",
        language="text",
    )
    st.caption(
        "Illustrative operating model only. Startup costs, tax, depreciation, "
        "financing, spoilage and unquoted local costs are excluded."
    )
    if finance_text:
        with st.expander("Financial Analyst task output"):
            st.write(finance_text)

with tabs[4]:
    st.subheader("Critical Review")
    st.write(
        "The reviewer receives research, finance, sustainability and writer "
        "outputs directly. Its review criteria include unsupported claims, "
        "source quality, arithmetic, pricing assumptions and missing risks."
    )
    if review_text:
        st.markdown("#### Skeptical Reviewer output")
        st.write(review_text)
    else:
        st.info("Generate a launch brief to run and view the reviewer’s output.")

with tabs[5]:
    st.subheader("Final Launch Brief")
    if brief_text:
        st.markdown(brief_text)
        st.download_button(
            "Download final_brief.md",
            data=brief_text,
            file_name="final_brief.md",
            mime="text/markdown",
            key="download_final_brief_tab",
        )
    else:
        st.info("No saved brief yet. Select “Generate Launch Brief” to create one.")

with tabs[6]:
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:600;margin-bottom:0.5rem">'
        "Project Reflection</div>",
        unsafe_allow_html=True,
    )
    if reflection_text:
        render_markdown_without_heading_anchors(reflection_text)
        st.download_button(
            "Download reflection.md",
            data=reflection_text,
            file_name="reflection.md",
            mime="text/markdown",
            key="download_reflection",
        )
    else:
        st.info("reflection.md is not present in the project folder.")
