"""
Run the sql-support-bot-evals dataset through the live agent and score it.

Usage:
    uv run python eval/run_eval.py
"""

from dotenv import load_dotenv
from langsmith.evaluation import evaluate

from agent import build_model, create_agent, message_text
from eval.dataset import DATASET_NAME
from eval.evaluators import dispatch
from eval.rate_limit import shared_rate_limiter

load_dotenv()

# Each example is run this many times. Scores here are noisy — the same input
# can pass one run and fail the next — so a pass *rate* across repetitions is
# far more meaningful than any single run's verdict.
NUM_REPETITIONS = 3

# Must track agent.py's model, or the evals grade something the app doesn't run.
# (The judge in evaluators.py deliberately stays on gpt-4o — a stable grader makes
# scores comparable across agent-model changes.)
_agent_model = build_model(rate_limiter=shared_rate_limiter, max_retries=6, timeout=60)
_agent = create_agent(model=_agent_model)


def target(inputs: dict) -> dict:
    """Replay the example's turn(s) through the agent, returning the final answer."""
    turns = inputs.get("conversation") or [inputs.get("question", "")]
    conversation_history = []
    ai_content = ""
    for turn in turns:
        conversation_history.append({"role": "user", "content": turn})
        result = _agent.invoke({"messages": conversation_history})
        ai_message = result["messages"][-1]
        ai_content = message_text(ai_message)
        conversation_history.append({"role": "assistant", "content": ai_content})
    return {"answer": ai_content}


if __name__ == "__main__":
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[dispatch],
        experiment_prefix="sql-support-bot",
        max_concurrency=1,
        num_repetitions=NUM_REPETITIONS,
    )
    print(results)
