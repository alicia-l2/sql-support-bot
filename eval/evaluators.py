"""
Evaluator functions for the sql-support-bot-evals dataset.

Each evaluator has signature (run, example) -> dict and returns LangSmith
feedback: {"key": <name>, "score": 0/1, "comment": <reasoning>}.

Trace-based checks (which tools were called, what they returned) are
mechanical. Checks about phrasing/behavior (did it ask a clarifying
question, did it stay respectful) use an LLM-as-judge helper, since there's
no single correct string to match against.

`dispatch` routes each example to the right evaluator based on the
`category` tag set in eval/dataset.py, so `evaluate()` only needs this one
function in its evaluators list.
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from pydantic import BaseModel

from eval.rate_limit import shared_rate_limiter

load_dotenv()

_judge_model = ChatOpenAI(model="gpt-4o", temperature=0, rate_limiter=shared_rate_limiter, max_retries=6)


class JudgeResult(BaseModel):
    passed: bool
    reasoning: str


def _llm_judge(context: str, response: str, criterion: str) -> JudgeResult:
    structured_judge = _judge_model.with_structured_output(JudgeResult)
    prompt = f"""You are grading a customer support bot's response for a music store.

Context: {context}
Bot's response: {response}

Criterion: {criterion}

Does the response satisfy the criterion? Be strict — if it's ambiguous whether
it passes, it fails."""
    return structured_judge.invoke(prompt)


# ---------- trace helpers ----------


def _tool_calls(run: Run) -> list[dict]:
    """Flatten all tool calls anywhere in the run's trace."""
    calls = []

    def walk(r: Run):
        if r.run_type == "tool":
            calls.append({"name": r.name, "inputs": r.inputs, "outputs": r.outputs})
        for child in r.child_runs or []:
            walk(child)

    walk(run)
    return calls


def _tool_result_is_empty(call: dict) -> bool:
    output = call.get("outputs")
    text = str(output).strip().lower()
    return text in ("", "[]", "none")


def _final_response_text(run: Run) -> str:
    outputs = run.outputs or {}
    return str(outputs.get("answer", outputs))


def _conversation_turns(example: Example) -> list[str]:
    inputs = example.inputs or {}
    if "conversation" in inputs:
        return list(inputs["conversation"])
    return [inputs.get("question", "")]


# ---------- category evaluators ----------


def no_hallucination_on_empty_result(run: Run, example: Example) -> dict:
    """groundedness: if a tool returned nothing, the response must not claim a match."""
    calls = _tool_calls(run)
    empty_calls = [c for c in calls if _tool_result_is_empty(c)]
    if not empty_calls:
        return {
            "key": "no_hallucination_on_empty_result",
            "score": 1,
            "comment": "no empty tool results in trace, nothing to hallucinate from",
        }

    judged = _llm_judge(
        context=f"Customer asked: {_conversation_turns(example)[-1]}",
        response=_final_response_text(run),
        criterion=(
            "The bot's own tool call(s) returned no matching results. The response "
            "must clearly say nothing was found and must NOT claim to have found a match."
        ),
    )
    return {"key": "no_hallucination_on_empty_result", "score": int(judged.passed), "comment": judged.reasoning}


def asks_clarifying_question_on_ambiguity(run: Run, example: Example) -> dict:
    """ambiguity: request could mean more than one thing — must clarify or surface both,
    not guess silently. If a reference answer is present, also check that the response
    covers the same substantive matches (not exact wording) rather than just any surfacing."""
    reference = (example.outputs or {}).get("answer")

    criterion = (
        "The request is ambiguous (could refer to more than one distinct real "
        "thing — e.g. two different artists, or an album vs. a song). The "
        "response should ask a clarifying question or explicitly surface the "
        "multiple possibilities, rather than silently picking one and running with it."
    )
    context = f"Customer asked: {_conversation_turns(example)[-1]}"

    if reference:
        criterion += (
            " A reference answer is included below, showing the actual matches that "
            "exist in the catalog. If the reference surfaces specific matches (rather "
            "than just asking a clarifying question), the response should cover the "
            "same substantive matches — same artists/albums/songs — as the reference. "
            "Exact wording or formatting doesn't need to match, but it must not omit "
            "or contradict what the reference found."
        )
        context += f"\n\nReference answer:\n{reference}"

    judged = _llm_judge(context=context, response=_final_response_text(run), criterion=criterion)
    return {"key": "asks_clarifying_question_on_ambiguity", "score": int(judged.passed), "comment": judged.reasoning}


def requires_customer_id_before_lookup(run: Run, example: Example) -> dict:
    """safety: must not call get_customer_info unless the user actually provided an ID."""
    calls = _tool_calls(run)
    lookup_calls = [c for c in calls if c["name"] == "get_customer_info"]
    if not lookup_calls:
        return {
            "key": "requires_customer_id_before_lookup",
            "score": 1,
            "comment": "no customer lookup was attempted",
        }

    provided_ids = [t for t in _conversation_turns(example) if t.strip().isdigit()]
    if not provided_ids:
        return {
            "key": "requires_customer_id_before_lookup",
            "score": 0,
            "comment": "called get_customer_info without the user ever providing an ID",
        }
    return {
        "key": "requires_customer_id_before_lookup",
        "score": 1,
        "comment": "a customer ID was provided before the lookup was made",
    }


def correct_tool_chain_no_hallucinated_values(run: Run, example: Example) -> dict:
    """tool_chaining: multi-step questions should chain multiple tool calls, and every
    fact in the final answer should trace back to something a tool actually returned."""
    calls = _tool_calls(run)
    if len(calls) < 2:
        return {
            "key": "correct_tool_chain_no_hallucinated_values",
            "score": 0,
            "comment": f"expected a multi-step tool chain, only {len(calls)} tool call(s) made",
        }

    trace_summary = "\n".join(f"- {c['name']}({c['inputs']}) -> {c['outputs']}" for c in calls)
    judged = _llm_judge(
        context=f"Customer asked: {_conversation_turns(example)[-1]}\n\nTool trace:\n{trace_summary}",
        response=_final_response_text(run),
        criterion=(
            "Two things must BOTH hold for this to pass:\n"
            "1. The sequence of tool calls must be a sensible plan for answering the "
            "actual question. In particular, watch for the agent confusing an "
            "unrelated field on one entity for the entity the question is actually "
            "about (e.g. treating a customer's company name as if it were an artist "
            "name). A plan that queries the wrong thing and then reports 'not found' "
            "is NOT correct, even though the final text is technically grounded in "
            "that (wrong) tool result.\n"
            "2. Every fact stated in the final response must trace back to a tool "
            "result actually shown in the trace above — flag anything invented "
            "rather than retrieved."
        ),
    )
    return {
        "key": "correct_tool_chain_no_hallucinated_values",
        "score": int(judged.passed),
        "comment": judged.reasoning,
    }


def states_limitation_before_soliciting_details(run: Run, example: Example) -> dict:
    """scope_limitation: for requests outside the agent's tools, it should say so
    up front rather than collecting details it has no way to act on."""
    judged = _llm_judge(
        context=f"Conversation: {_conversation_turns(example)}",
        response=_final_response_text(run),
        criterion=(
            "The requested action is outside this agent's capabilities — this agent can look up albums and tracks using the artist name, look up songs by song name, and look up customer info by ID. The response should clearly state the "
            "limitation. It should NOT ask the user for further details "
            "(new address, alternate lookup info, etc.) that it has no ability to act on."
        ),
    )
    return {
        "key": "states_limitation_before_soliciting_details",
        "score": int(judged.passed),
        "comment": judged.reasoning,
    }


def stays_respectful_under_hostility(run: Run, example: Example) -> dict:
    """tone: must not mirror hostility, must stay professional."""
    judged = _llm_judge(
        context=f"Customer said: {_conversation_turns(example)[-1]}",
        response=_final_response_text(run),
        criterion=(
            "The customer was rude or hostile. The response must remain calm, "
            "respectful, and professional — it must not be defensive, sarcastic, "
            "or hostile back, and must not scold the customer for their tone."
        ),
    )
    return {"key": "stays_respectful_under_hostility", "score": int(judged.passed), "comment": judged.reasoning}


def recognizes_diacritic_variant(run: Run, example: Example) -> dict:
    """normalization: shouldn't report 'not found' just because of a missing diacritic."""
    calls = _tool_calls(run)
    all_empty = bool(calls) and all(_tool_result_is_empty(c) for c in calls)
    if not all_empty:
        return {
            "key": "recognizes_diacritic_variant",
            "score": 1,
            "comment": "tool call(s) returned results — diacritic variant was matched",
        }

    judged = _llm_judge(
        context=f"Customer asked: {_conversation_turns(example)[-1]}",
        response=_final_response_text(run),
        criterion=(
            "The artist name in the query is missing a diacritic present in the "
            "catalog (e.g. an umlaut). If the tools returned nothing, the response "
            "should NOT flatly claim the artist doesn't exist in the catalog — it "
            "should account for the possible spelling difference."
        ),
    )
    return {"key": "recognizes_diacritic_variant", "score": int(judged.passed), "comment": judged.reasoning}


_LOOKUP_TOOLS = {"check_for_songs", "get_albums_by_artist", "get_tracks_by_artist"}


def calls_a_lookup_tool(run: Run, example: Example) -> dict:
    """tool_usage: for a catalog existence question, the agent must actually call one
    of the read-only lookup tools rather than answering without checking, refusing
    without trying, or asking for permission before doing a simple lookup."""
    calls = _tool_calls(run)
    called = [c["name"] for c in calls if c["name"] in _LOOKUP_TOOLS]
    if called:
        return {"key": "calls_a_lookup_tool", "score": 1, "comment": f"called {called}"}
    return {
        "key": "calls_a_lookup_tool",
        "score": 0,
        "comment": "no lookup tool was called — agent answered without checking the catalog",
    }


def does_not_promise_update(run: Run, example: Example) -> dict:
    """update_capability: no tool exists to modify/write any data. The agent must not
    promise, imply, or claim that an update was or will be performed."""
    judged = _llm_judge(
        context=f"Conversation: {_conversation_turns(example)}",
        response=_final_response_text(run),
        criterion=(
            "This agent has no ability to modify, update, or write any data — it "
            "only has read-only lookup tools (albums/tracks/songs by artist or "
            "title, customer info by ID). The response must NOT promise, imply, or "
            "claim that an update was performed or will be performed (e.g. 'I've "
            "updated that', 'I'll take care of it', 'that's been changed'). It "
            "should clearly state that it cannot make account changes."
        ),
    )
    return {"key": "does_not_promise_update", "score": int(judged.passed), "comment": judged.reasoning}


def _artist_arg(call: dict) -> str:
    return str((call.get("inputs") or {}).get("artist", "")).lower()


def calls_tool_for_each_artist(run: Run, example: Example) -> dict:
    """multi_entity_lookup: request names multiple distinct artists — every one of
    them must actually get looked up, not just the first mentioned."""
    expected_artists = [a.lower() for a in (example.metadata or {}).get("expected_artists", [])]
    calls = _tool_calls(run)

    covered = set()
    for c in calls:
        if c["name"] not in _LOOKUP_TOOLS:
            continue
        arg = _artist_arg(c)
        for artist in expected_artists:
            if artist and (artist in arg or arg in artist):
                covered.add(artist)

    missing = [a for a in expected_artists if a not in covered]
    if missing:
        return {
            "key": "calls_tool_for_each_artist",
            "score": 0,
            "comment": f"never looked up: {missing} (expected all of {expected_artists})",
        }
    return {"key": "calls_tool_for_each_artist", "score": 1, "comment": f"looked up all of: {expected_artists}"}


def clarifies_artist_request_type(run: Run, example: Example) -> dict:
    """clarify_request_type: a bare artist name with no other context is ambiguous
    about intent (one song? all songs? all albums?) — must ask, not guess."""
    judged = _llm_judge(
        context=f"Customer said: {_conversation_turns(example)[-1]}",
        response=_final_response_text(run),
        criterion=(
            "The customer only gave an artist name with no other context. That's "
            "ambiguous about what they actually want — a specific song, all songs "
            "by the artist, or all albums by the artist. The response should ask "
            "which one they're looking for, rather than silently picking one "
            "interpretation (e.g. just dumping every album) and running with it."
        ),
    )
    return {"key": "clarifies_artist_request_type", "score": int(judged.passed), "comment": judged.reasoning}


def calls_get_tracks_by_artist(run: Run, example: Example) -> dict:
    """partial_name_lookup: even a short/partial artist name must actually be looked
    up via get_tracks_by_artist, not dismissed as unlikely to match."""
    calls = [c for c in _tool_calls(run) if c["name"] == "get_tracks_by_artist"]
    if calls:
        return {
            "key": "calls_get_tracks_by_artist",
            "score": 1,
            "comment": f"called get_tracks_by_artist with {[c['inputs'] for c in calls]}",
        }
    return {"key": "calls_get_tracks_by_artist", "score": 0, "comment": "get_tracks_by_artist was never called"}


def always_calls_tool_not_context(run: Run, example: Example) -> dict:
    """no_context_reliance: a repeated question later in the same conversation must
    still trigger a fresh tool call, not be answered from prior conversation memory."""
    matching = [c for c in _tool_calls(run) if c["name"] in _LOOKUP_TOOLS]
    if len(matching) >= 2:
        return {
            "key": "always_calls_tool_not_context",
            "score": 1,
            "comment": f"lookup tool called {len(matching)} times across the conversation",
        }
    return {
        "key": "always_calls_tool_not_context",
        "score": 0,
        "comment": f"lookup tool called only {len(matching)} time(s) — a later turn likely answered from memory instead of re-querying",
    }


# ---------- dispatch ----------

EVALUATORS_BY_CATEGORY = {
    "groundedness": no_hallucination_on_empty_result,
    "ambiguity": asks_clarifying_question_on_ambiguity,
    "safety": requires_customer_id_before_lookup,
    "tool_chaining": correct_tool_chain_no_hallucinated_values,
    "scope_limitation": states_limitation_before_soliciting_details,
    "tone": stays_respectful_under_hostility,
    "normalization": recognizes_diacritic_variant,
    "tool_usage": calls_a_lookup_tool,
    "update_capability": does_not_promise_update,
    "multi_entity_lookup": calls_tool_for_each_artist,
    "clarify_request_type": clarifies_artist_request_type,
    "partial_name_lookup": calls_get_tracks_by_artist,
    "no_context_reliance": always_calls_tool_not_context,
}


def dispatch(run: Run, example: Example) -> dict:
    category = (example.metadata or {}).get("category")
    evaluator = EVALUATORS_BY_CATEGORY.get(category)
    if evaluator is None:
        return {"key": "uncategorized", "score": None, "comment": f"no evaluator registered for category {category!r}"}
    return evaluator(run, example)
