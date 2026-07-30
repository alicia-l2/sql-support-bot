"""
Run the sql-support-bot-evals dataset through the live agent and score it.

Usage:
    uv run python eval/run_eval.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith.evaluation import evaluate

from agent import create_agent
from eval.dataset import DATASET_NAME
from eval.evaluators import dispatch
from eval.rate_limit import shared_rate_limiter

load_dotenv()

_agent_model = ChatOpenAI(model="gpt-4o", temperature=0, rate_limiter=shared_rate_limiter, max_retries=6)
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
        ai_content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        conversation_history.append({"role": "assistant", "content": ai_content})
    return {"answer": ai_content}


if __name__ == "__main__":
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[dispatch],
        experiment_prefix="sql-support-bot",
        max_concurrency=1,
    )
    print(results)
