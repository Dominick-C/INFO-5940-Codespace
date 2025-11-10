# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# === TESTING KEY OVERRIDES (DO NOT COMMIT REAL SECRETS) ======================
# For local testing only: hard-code keys here if you don't want to use env/secrets.
# Leave as None to fall back to environment variables or Streamlit secrets.
TEST_KEYS: Dict[str, Optional[str]] = {
    "OPENAI_API_KEY": "put opean ai key here",  # e.g., "sk-..."; set only for local testing
    "TAVILY_API_KEY": "put tavily key here",  # e.g., "tvly-dev-..."; set only for local testing
}

def get_secret(name: str) -> Optional[str]:
    """
    Unified secret getter:
    1) TEST_KEYS override (for local testing only)
    2) Environment variables
    3) Streamlit secrets (if configured)
    """
    override = TEST_KEYS.get(name)
    if override:
        return override
    env_val = os.environ.get(name)
    if env_val:
        return env_val
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]  # type: ignore[index]
    return None

# If testing overrides are set, mirror them into environment so SDKs (e.g., OpenAI) see them.
for _k, _v in TEST_KEYS.items():
    if _v:
        os.environ[_k] = _v
# ============================================================================

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from TEST_KEYS/env/Streamlit secrets via get_secret().
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = get_secret("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY (set in TEST_KEYS, env, or st.secrets)."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """
Role: You are the Reviewer Agent. You MUST validate the Planner’s itinerary before it is shown to the user.
You MAY (and should) call the `internet_search` tool to fact-check specific claims.

Inputs:
- The Planner’s Markdown itinerary (summary, budget, city clusters, day-by-day plan).

Goals:
1) Feasibility & consistency
   • Opening hours / last entry times (by day/season if possible)
   • Ticket requirements/prices/availability (note if advance booking is needed)
   • Travel-time realism (within-city and inter-city transfers)
   • Pace realism (avoid unrealistic stacking; add buffers)
   • Overlaps/conflicts (time collisions, closed venues, too-tight connections)
   • Budget realism (flag overages; suggest swaps that keep interests intact)
2) Produce specific, actionable fixes as a **Delta List** (precise edits, not generalities).
3) Apply those deltas to output a **Corrected Itinerary** ready for the user.

How to use the tool:
- Call `internet_search("concise query")` when evidence is required (e.g., "Louvre last entry time", "Rome→Florence train time", "Sagrada Familia ticket price").
- Prefer official or authoritative sources by title/snippet. If results are inconclusive, state uncertainty and keep conservative assumptions—do not fabricate specifics.

Output format (Markdown only):
## Validation Summary
- Bullet points of what you verified and any risks/uncertainties (quote source titles/snippets briefly).

## Delta List
1) Day X — Change: <precise replacement/move>. Reason: <why>. Evidence: <source title>.
2) ...

## Corrected Itinerary
- Reprint the itinerary with all fixes applied. If no changes are required, say so and reprint the original.

## Notes & Caveats
- Prebooking windows, seasonal schedules, strike/holiday risks, or anything the traveler should double-check.

Ground rules:
- Maintain the user’s interests, dates, and budget while making corrections.
- Add ~15–30 minute buffers around major entries and inter-neighborhood moves.
- Keep claims modest if the tool cannot confirm specifics.
"""

PLANNER_INSTRUCTIONS = """
Role: You are the Planner Agent. You must NOT use the internet.
Task: Expand the user’s vague travel prompt into a clear, feasible day-by-day itinerary.

Requirements:
- Respect user constraints (dates/timing if given, total budget, interests, preferred pacing).
- Cluster days into sensible city/area groupings to minimize transit overhead.
- For each day, include: morning/afternoon/evening blocks with approximate times, activity names, location/area, estimated costs, and simple logistics (walk/metro/bus/train/ridehail).
- Keep a running budget (daily subtotal + trip total). Use conservative ranges when uncertain.
- Do NOT assert precise opening hours or exact ticket prices; mark any item needing confirmation with **TO_VERIFY** for the Reviewer.
- Prefer walking or single-line transit; avoid unrealistic back-to-back hops across a city.

Output format (Markdown only):
# Trip Summary
- 2–4 sentences on theme, pacing, and city clusters.

## Budget Overview
- Currency (default USD if unspecified), daily subtotals, and total estimate.

## City Cluster Plan
- Cities/areas in order with 1–2 sentences on why the clustering reduces transit time.

## Day-by-Day Itinerary
Day 1 — <City> (<date if provided>)
- 09:00–10:30: <Activity @ Place> — ~$<est>  — Logistics: <walk/metro/etc>
- 11:00–12:30: <Activity> — ~$<est>
- 12:30–14:00: Lunch in <area> — ~$<est>
- 14:30–16:30: <Activity> — ~$<est>  **TO_VERIFY** (hours/tickets)
- 17:00–19:00: <Light activity / neighborhood stroll> — ~$<est or $0>
- Daily subtotal: ~$<sum>
(Repeat for each day.)

## Assumptions & Items to Verify
- Key assumptions (e.g., staying near city center, transit pass).
- Bullet list of all **TO_VERIFY** items for the Reviewer.

Tone & style:
- Practical and readable.
- Approximate times and even pacing.
- Conservative cost ranges; avoid overpacking days.
"""

reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]  # enable the provided Tavily-backed search tool
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
)
# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("✅ Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("🤖 **Reviewer Agent** (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
