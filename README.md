# Campus Café launch-team agents

A CrewAI workflow and Streamlit dashboard for an evidence-led campus café pilot. The existing five-agent team uses LangChain tools for web search and arithmetic, and saves the reviewer's corrected output to `final_brief.md`. The accompanying `reflection.md` describes the collaboration.

## Run the Streamlit application

Requires Python 3.10+ and API keys for Groq and Tavily Search. From PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Set GROQ_API_KEY and TAVILY_API_KEY in .env
streamlit run .\streamlit_app.py
```

The app's **Generate Launch Brief** button runs the existing CrewAI workflow, shows the agent task outputs and clickable source URLs, and writes the reviewed result to `final_brief.md`. It also offers downloads of `final_brief.md` and `reflection.md`. If search or another tool fails, the app reports the error and does not replace an existing brief.

## Run the workflow from the command line

```powershell
python .\cafe_launch.py
```

The dependencies include CrewAI's LiteLLM extra, which enables the `groq/` provider, and the standalone `langchain-tavily` integration (instead of the deprecated `langchain-community` wrapper). LiteLLM is pinned to a version tested with CrewAI's MCP SDK dependency to avoid an incompatible MCP import. Keep this pin unless the CrewAI/LiteLLM/MCP dependency set is upgraded and tested together. The default model is `groq/qwen/qwen3.8-27b`; set `LLM_MODEL` in `.env` to another Groq model supported by CrewAI if desired. The default orchestration is sequential; `python .\cafe_launch.py --process hierarchical` runs CrewAI's hierarchical process using the configured model as manager.

## Design and safeguards

- Every agent has a role, goal and backstory. The writer explicitly receives research, finance and sustainability task outputs; the reviewer explicitly receives all four upstream outputs.
- The Market Researcher and Sustainability Officer use a Tavily search tool built with LangChain; the Financial Analyst and Skeptical Reviewer use a LangChain calculator. CrewAI adapters expose those LangChain tools to the agents.
- The financial inputs are centralized in `ASSUMPTIONS` in `cafe_launch.py`. They are intentionally illustrative and should be replaced with campus quotes and local demand data.
- Research prompts require source URLs, dates, population scope and evidence-gap disclosures. The writer/reviewer must cite sourced claims and label assumptions.
- Missing credentials, unavailable search, or arithmetic errors are surfaced; the workflow does not silently substitute fabricated evidence or calculations. A failed tool call aborts the run, so no new final brief is written.
- Groq does not accept CrewAI's internal `cache_breakpoint` message marker. `GroqLLM` removes that marker before forwarding messages while preserving message content and all other fields.
- Keep real credentials in the ignored local `.env` file only. If a key is exposed, revoke it at its provider and replace it before running the workflow.

## Process choice

Sequential is the default because the writer should receive the completed research, financial worksheet and sustainability audit before drafting. Hierarchical mode is exposed for comparison; it adds manager-led delegation, which can be useful for exploratory task allocation but may add overhead and make the information flow less predictable. Compare the two by checking citation coverage, arithmetic, unsupported-claim count, final word count and runtime over repeated runs.
