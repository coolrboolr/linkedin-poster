from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import sys
from src.state import AppState
from src.agents import (
    scan_trending_topics,
    fetch_arxiv_papers,
    rank_papers,
    conversation_node,
    write_post,
    human_approval,
    update_memory,
    load_memory,
    human_paper_review,
)
from src.services.logger import get_logger

logger = get_logger(__name__)

# Planning Router
def _has_pending_user_message(state: AppState) -> bool:
    if not state.chat_history:
        return False
    last = state.chat_history[-1]
    if last.get("role") != "user":
        return False
    message = (last.get("message") or "").strip()
    if not message:
        return False
    return f"User: {message}" not in state.clarification_history

async def planning_router(state: AppState) -> dict:
    """
    Routes through the planning phase.

    Hard guarantees:
      - trend_scanner runs before anything else
      - arxiv_fetcher runs after that

    After those bootstrap steps, we are in a planning loop that can:
      - run conversation_agent to confirm focus before ranking
      - rank papers
      - run conversation_agent until the user explicitly accepts the plan
      - request human paper review to confirm/switch the article
      - exit to execution_router once the user is ready AND the paper is approved
    """
    if state.exit_requested:
        logger.info("Exit requested during planning; routing to END.")
        return {"next_step": "exit"}

    if state.awaiting_user_response:
        if _has_pending_user_message(state):
            logger.info("Pending user response detected; resuming conversation.")
            return {"next_step": "conversation_agent", "awaiting_user_response": False}
        logger.info("Awaiting user response; routing to conversation_agent to capture resume.")
        return {"next_step": "conversation_agent"}

    # 1. Bootstrap: enforce trend_scanner then arxiv_fetcher
    if not state.trending_keywords:
        next_step = "trend_scanner"
    elif not state.paper_candidates:
        next_step = "arxiv_fetcher"
    elif not state.selected_paper:
        # We have topics and papers; confirm focus with the user before ranking
        if not state.user_ready:
            next_step = "conversation_agent"
        else:
            next_step = "relevance_ranker"
    elif not state.user_ready:
        # Discuss with the user until they explicitly accept the plan
        next_step = "conversation_agent"
    elif not state.paper_approved:
        # User is ready; confirm or switch the article
        next_step = "human_paper_review"
    else:
        # Ready and article approved — move to execution
        next_step = "execution_router"

    logger.info(f"Planning router decided: {next_step}")
    return {"next_step": next_step}

# Execution Router
async def execution_router(state: AppState) -> dict:
    """
    Routes through the execution phase (Write -> Approve -> Memory).
    """
    if state.exit_requested:
        logger.info("Exit requested during execution; routing to END.")
        return {"next_step": "exit", "return_to_conversation": False}

    next_step = "memory_updater"  # Default end state

    if state.return_to_conversation:
        # Hop back into planning loop to continue discussion
        next_step = "planning_router"
    elif state.revision_requested or not state.post_draft:
        # No draft yet, or human requested edits → regenerate
        next_step = "post_writer"
    elif not state.approved:
        # Draft exists; seek approval or further feedback
        next_step = "human_approval"
    else:
        next_step = "memory_updater"

    logger.info(f"Execution router decided: {next_step}")
    return {"next_step": next_step, "return_to_conversation": False}

# Conditional edge function
def get_next_step(state: AppState):
    return state.next_step

# Define the graph
workflow = StateGraph(AppState)

# Add nodes
workflow.add_node("load_memory", load_memory)
workflow.add_node("planning_router", planning_router)
workflow.add_node("execution_router", execution_router)

workflow.add_node("trend_scanner", scan_trending_topics)
workflow.add_node("arxiv_fetcher", fetch_arxiv_papers)
workflow.add_node("relevance_ranker", rank_papers)
workflow.add_node("human_paper_review", human_paper_review)
workflow.add_node("conversation_agent", conversation_node)

workflow.add_node("post_writer", write_post)
workflow.add_node("human_approval", human_approval)
workflow.add_node("memory_updater", update_memory)

# Edges
# Start -> Load Memory -> Planning Router
workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "planning_router")

# Planning Phase Logic
workflow.add_conditional_edges(
    "planning_router",
    get_next_step,
    {
        "trend_scanner": "trend_scanner",
        "arxiv_fetcher": "arxiv_fetcher",
        "relevance_ranker": "relevance_ranker",
        "human_paper_review": "human_paper_review",
        "conversation_agent": "conversation_agent",
        "execution_router": "execution_router",
        "exit": END,
    }
)

# Planning Nodes -> Planning Router
workflow.add_edge("trend_scanner", "planning_router")
workflow.add_edge("arxiv_fetcher", "planning_router")
workflow.add_edge("relevance_ranker", "planning_router")
workflow.add_edge("human_paper_review", "planning_router")
workflow.add_edge("conversation_agent", "planning_router")

# Execution Phase Logic
workflow.add_conditional_edges(
    "execution_router",
    get_next_step,
    {
        "post_writer": "post_writer",
        "human_approval": "human_approval",
        "memory_updater": "memory_updater",
        "planning_router": "planning_router",
        "exit": END,
    }
)

# Execution Nodes -> Execution Router
workflow.add_edge("post_writer", "execution_router")
workflow.add_edge("human_approval", "execution_router")

# Memory Updater -> End
workflow.add_edge("memory_updater", END)

# Compile with in-memory checkpointing unless LangGraph API is managing persistence.
use_checkpointer = "langgraph_api" not in sys.modules
checkpointer = MemorySaver() if use_checkpointer else None
graph = workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()

if __name__ == "__main__":
    import asyncio
    async def main():
        print("Starting Graph...")
        config = {"configurable": {"thread_id": "1"}}
        async for output in graph.astream(AppState(), config=config):
            for key, value in output.items():
                print(f"Finished node: {key}")
    
    asyncio.run(main())
