import asyncio
import sys
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(".")

from src.agents.conversation_agent import conversation_node
from src.state import AppState

async def verify_conversation_flow():
    print("--- Starting Verification of Conversation Agent ---")
    
    # 1. Setup Mock State
    initial_state = AppState(
        trending_keywords=["AI Agents", "LLMs"],
        paper_candidates=[{"title": "Paper A", "summary": "Summary A"}],
        selected_paper={"title": "Paper A", "summary": "Summary A"},
        chat_history=[],
        clarification_history=[]
    )
    
    # 2. Mock Dependencies
    with patch("src.agents.conversation_agent.interrupt") as mock_interrupt, \
         patch("src.agents.conversation_agent.settings") as mock_settings, \
         patch("src.agents.conversation_agent._invoke_legacy") as mock_invoke_legacy:

        # Configure Settings
        mock_settings.openai_api_key = "fake-key"
        # We ensure _should_use_tools returns False or we just rely on _invoke_legacy being called
        # The code checks `not isinstance(llm, MagicMock)` for tools, so if we mock init_chat_model to return a MagicMock...
        # But we are mocking _invoke_legacy. 
        # We need to make sure the code calls _invoke_legacy.
        # This happens if tool_ready is False.
        # tool_ready requires llm.bind_tools AND not MagicMock.
        # We can just let init_chat_model be real or mock? 
        # Actually conversation_node calls init_chat_model. We should mock that too to avoid real API calls.
        
        with patch("src.agents.conversation_agent.init_chat_model") as mock_init_model:
            mock_llm = MagicMock()
            mock_init_model.return_value = mock_llm
            # This ensures tool_ready is False because isinstance(llm, MagicMock) is True.
            
            # Mock _invoke_legacy return values
            # It returns (content, angles, question)
            mock_invoke_legacy.side_effect = [
                ("Content 1", ["Angle A"], "Question 1: What topic?"),
                ("Content 2", [], "Question 2: Does that help?")
            ]

            # --- RUN 1 (Initial Assistant Question) ---
            print("\n[Step 1] Running conversation_node (Initial)...")
            
            # User asks a question in response
            mock_interrupt.return_value = {
                "type": "response",
                "args": "But how do AI Agents handle errors?" 
            }

            updates_1 = await conversation_node(initial_state)
            
            print("Updates from Run 1:", updates_1)
            
            # Verify the user message was added to the history in the updates
            last_msg = updates_1["chat_history"][-1]
            assert last_msg["role"] == "user"
            assert last_msg["message"] == "But how do AI Agents handle errors?"
            print("✅ User question successfully added to chat history.")

            # --- RUN 2 (Agent Answering) ---
            print("\n[Step 2] Running conversation_node (Response to User Question)...")
            
            # Update state
            state_2 = initial_state.model_copy(update=updates_1)
            
            # Second user response: Accept
            mock_interrupt.return_value = {"type": "accept", "args": "Yes, perfect."}

            await conversation_node(state_2)
            
            # Check inputs passed to _invoke_legacy in the SECOND call
            # call_args_list[1] is the second call
            clean_args = mock_invoke_legacy.call_args_list[1]
            # Signature: _invoke_legacy(prompt_text: str, inputs: Dict[str, Any])
            # args[0] is prompt_text, args[1] is inputs
            inputs = clean_args[0][1]
            
            history_text = inputs["history"]
            print(f"\nHistory Text sent to LLM:\n---\n{history_text}\n---")
            
            if "But how do AI Agents handle errors?" in history_text:
                print("✅ User question was present in the LLM context (history input).")
            else:
                print("❌ User question was NOT found in the LLM context.")
                raise AssertionError("User question missing from history!")

if __name__ == "__main__":
    asyncio.run(verify_conversation_flow())
