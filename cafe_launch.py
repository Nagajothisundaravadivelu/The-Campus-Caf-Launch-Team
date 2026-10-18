"""Generate a sourced campus-cafe launch brief with CrewAI and LangChain tools."""

from __future__ import annotations

import argparse
import ast
import operator
import os
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool as CrewBaseTool
from dotenv import load_dotenv
from langchain_core.tools import BaseTool as LangChainBaseTool
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from pydantic import ConfigDict, Field


class FailingTool(CrewBaseTool):
    """Test-only tool that simulates a tool outage for fail-safe checks."""

    name: str = "failing_tool"
    description: str = "Always raises an error to simulate a failed tool call."

    def _run(self, query: str) -> str:
        raise RuntimeError(f"Simulated tool failure while handling query: {query}")


class GroqLLM(LLM):
    """Remove CrewAI's internal prompt-cache marker before sending Groq requests."""

    def _format_messages_for_provider(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        clean_messages = []
        for message in messages:
            clean_message = dict(message)
            clean_message.pop("cache_breakpoint", None)
            clean_messages.append(clean_message)
        return super()._format_messages_for_provider(clean_messages)


ASSUMPTIONS = {
    "enrolled_students": 8000,
    "operating_days_per_month": 22,
    "average_transaction_inr": 120,
    "variable_cost_per_transaction_inr": 54,
    "monthly_space_cost_inr": 80000,
    "monthly_staff_cost_inr": 140000,
    "monthly_utilities_inr": 25000,
    "monthly_other_fixed_costs_inr": 15000,
    "target_transactions_per_day": 240,
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate(node.left), _evaluate(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numeric literals and +, -, *, /, parentheses are allowed.")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression without executing arbitrary code."""
    result = _evaluate(ast.parse(expression, mode="eval"))
    return f"{expression} = {result:,.2f}"


class LangChainToolAdapter(CrewBaseTool):
    """Expose a LangChain tool through CrewAI's tool interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    langchain_tool: LangChainBaseTool = Field(exclude=True)
    argument_name: str

    def _run(self, query: str) -> str:
        result = self.langchain_tool.invoke({self.argument_name: query})
        if isinstance(result, str):
            return result
        return str(result)


def build_crew(process_name: str, simulate_tool_failure: bool = False) -> Crew:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required to run the agents.")
    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is required for sourced web research.")

    model = os.getenv("LLM_MODEL", "groq/qwen/qwen3.8-27b")
    def make_llm() -> GroqLLM:
        return GroqLLM(model=model, max_tokens=900)

    search_tool = (
        FailingTool()
        if simulate_tool_failure
        else TavilySearch(max_results=2, search_depth="basic")
    )
    research_search = LangChainToolAdapter(
        name="LangChain web search",
        description=(
            "Search the web for reliable India-specific evidence. Include query "
            "terms for the claim, date and source type; return URLs and source dates."
        ),
        langchain_tool=search_tool,
        argument_name="query",
    )
    calculator_tool = StructuredTool.from_function(
        func=calculator,
        name="calculator",
        description=(
            "Evaluate arithmetic expressions using numeric literals and +, -, *, / "
            "and parentheses. Pass the complete expression as a string."
        ),
    )
    calculator_adapter = LangChainToolAdapter(
        name="LangChain calculator",
        description=(
            "Calculate and show arithmetic. Pass the expression in the query, "
            "for example: (80000 + 140000 + 25000 + 15000) / (120 - 54)."
        ),
        langchain_tool=calculator_tool,
        argument_name="expression",
    )

    market_researcher = Agent(
        role="Market Researcher",
        goal=(
            "Find current, verifiable facts on Indian student spending behavior, "
            "local café prices, and food/sustainability trends."
        ),
        backstory=(
            "You are a careful research analyst who distinguishes student-specific "
            "evidence from broad household or consumer proxies. You never invent a "
            "citation, figure, date, or local price."
        ),
        tools=[research_search],
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )
    financial_analyst = Agent(
        role="Financial Analyst",
        goal=(
            "Calculate a transparent monthly operating break-even and test the "
            "stated demand case using the calculator tool."
        ),
        backstory=(
            "You are a cautious campus-retail analyst. Every input is labelled as "
            "a user-provided fact, researched fact, or planning assumption. You "
            "show each equation and distinguish operating break-even from startup "
            "payback."
        ),
        tools=[calculator_adapter],
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )
    sustainability_officer = Agent(
        role="Sustainability Officer",
        goal=(
            "Verify environmental and plant-based claims and recommend practical, "
            "measurable pilot choices without overstating benefits."
        ),
        backstory=(
            "You review food-service sustainability claims for evidence, scope and "
            "measurability. You flag unsupported carbon, waste and demand claims."
        ),
        tools=[research_search],
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )
    marketing_writer = Agent(
        role="Marketing Writer",
        goal=(
            "Turn the evidence and financial model into a persuasive, concise "
            "one-page decision brief for the Dean."
        ),
        backstory=(
            "You write clear, credible recommendations for university leaders. "
            "You explain limitations plainly and do not hide commercial risks."
        ),
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )
    skeptical_reviewer = Agent(
        role="Skeptical Reviewer",
        goal=(
            "Audit the draft for unsupported claims, missing risks and arithmetic "
            "errors, then deliver the corrected final brief."
        ),
        backstory=(
            "You are an independent red-team editor. You verify each figure against "
            "the task outputs, remove weak claims, retain source links and produce "
            "a decision-ready final version no longer than one page."
        ),
        tools=[calculator_adapter, research_search],
        llm=make_llm(),
        verbose=False,
        allow_delegation=False,
        max_iter=2,
    )

    assumptions_text = "\n".join(
        f"- {key}: {value:,} (INR where the name specifies INR)"
        for key, value in ASSUMPTIONS.items()
    )
    research_task = Task(
        description=(
            "Search reliable sources for Indian student spending (label proxies), "
            "Indian café menu prices, and plant-based/sustainability trends. Return "
            "at most 3 bullets, each with one fact and URL. Label population/geo "
            "limits and evidence gaps. Do not guess. Output <=100 words."
        ),
        expected_output="Source-linked evidence bullets and gaps, <=100 words.",
        agent=market_researcher,
    )
    finance_task = Task(
        description=(
            f"Use these planning inputs (all are assumptions except student "
            f"enrollment, which is supplied in the scenario):\n{assumptions_text}\n"
            "Calculate with the calculator tool, showing the expression and result "
            "for: monthly fixed costs (sum the four cost lines), contribution per "
            "transaction, monthly break-even transactions (round up to a whole "
            "transaction), break-even transactions per operating day, daily "
            "break-even share of enrollment, monthly sales/cost/contribution at "
            "the target transaction count, and monthly operating surplus or "
            "shortfall. Label exclusions such as startup costs, tax, depreciation "
            "and financing. Never introduce an unlabelled numerical input."
        ),
        expected_output=(
            "Calculations with expressions and exclusions, <=120 words."
        ),
        agent=financial_analyst,
    )
    sustainability_task = Task(
        description=(
            "Use web search to verify one plant-based/sustainability trend. Give "
            "source URL, scope caveat and one measurable pilot suggestion. Do not "
            "claim environmental benefits without measurement. Output <=70 words."
        ),
        expected_output="Source-linked trend and caveat, <=70 words.",
        agent=sustainability_officer,
    )
    draft_task = Task(
        description=(
            "Draft a <=180-word Dean brief from the research, finance and "
            "sustainability task context. Cite spending evidence/proxy, café prices "
            "and trend evidence; disclose gaps. Show every break-even assumption "
            "and equation, recommendation and risks. Label or cite every number."
        ),
        expected_output="A cited, one-page decision brief draft.",
        agent=marketing_writer,
        context=[research_task, finance_task, sustainability_task],
    )
    final_task = Task(
        description=(
            "Independently audit the draft and upstream task outputs for "
            "unsupported claims, source quality and relevance, arithmetic, "
            "unvalidated pricing assumptions, and missing operational or market "
            "risks. Verify that each cited URL is present in the research outputs "
            "and that its scope supports the associated claim; remove weak or "
            "unsupported statistics rather than inventing replacements. Return "
            "only the final <=180-word brief. Cite each retained market fact "
            "with a Markdown URL already in the research outputs and state "
            "evidence gaps. "
            "Use these exact calculations from the supplied assumptions: fixed "
            "cost ₹260,000/month; contribution ₹66/order; monthly break-even "
            "3,940 orders; daily break-even 3,940 / 22 = 180 orders/day rounded "
            "up; daily enrollment share 180 / 8,000 = 2.25%; target case at 240 "
            "orders/day has monthly operating surplus "
            "(240 * 22 * 66) - 260000 = ₹88,480. Do not claim target surplus at "
            "break-even. Include risks and local validation. No invented dates, "
            "sources or numbers."
        ),
        expected_output="The final corrected one-page brief, ready for the Dean.",
        agent=skeptical_reviewer,
        context=[research_task, finance_task, sustainability_task, draft_task],
    )

    process = Process.sequential if process_name == "sequential" else Process.hierarchical
    crew = Crew(
        agents=[
            market_researcher,
            financial_analyst,
            sustainability_officer,
            marketing_writer,
            skeptical_reviewer,
        ],
        tasks=[
            research_task,
            finance_task,
            sustainability_task,
            draft_task,
            final_task,
        ],
        process=process,
        manager_llm=make_llm() if process == Process.hierarchical else None,
        verbose=False,
    )
    return crew


def run_workflow(
    process_name: str = "sequential",
    output_path: Path | str = Path("final_brief.md"),
) -> Any:
    """Run the launch-team workflow and persist its reviewed brief."""
    result = build_crew(process_name).kickoff()
    destination = Path(output_path)
    destination.write_text(str(result).strip() + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process",
        choices=("sequential", "hierarchical"),
        default="sequential",
        help="CrewAI task orchestration mode (default: sequential).",
    )
    parser.add_argument(
        "--simulate-tool-failure",
        action="store_true",
        help="Force a failing search tool to demonstrate graceful failure behavior.",
    )
    args = parser.parse_args()
    output_path = Path("final_brief.md")
    if args.simulate_tool_failure:
        result = build_crew(
            args.process, simulate_tool_failure=True
        ).kickoff()
        output_path.write_text(str(result).strip() + "\n", encoding="utf-8")
    else:
        run_workflow(args.process, output_path)
    print(f"Saved reviewed brief to {output_path.resolve()}")


if __name__ == "__main__":
    main()
