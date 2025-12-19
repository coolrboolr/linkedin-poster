import pytest
import shutil
import json
from unittest.mock import MagicMock, patch
from src.state import AppState
from src.agents.human_approval import human_approval
from src.agents.memory_updater import update_memory
from src.agents.post_writer import write_post
from src.core.paths import MEMORY_DIR

@pytest.mark.asyncio
async def test_multi_iteration_edit_persistence():
    """
    Simulate a chain of:
    1. Writer generates Draft 1.
    2. Approval: User says "Edit" -> Draft 2 (manual).
    3. Writer skipped (since user edited manually), but Approval sets state.
    4. Approval: User says "Response" -> "Make it shorter".
    5. Writer generates Draft 3 (using instruction).
    6. Approval: User says "Accept".
    7. Memory Updater runs.
    
    Verify:
    - Edit requests are accumulated.
    - Final memory contains feedback from the cycle.
    """
    
    # Setup temporary memory dir
    test_memory_dir = MEMORY_DIR / "test_run"
    if test_memory_dir.exists():
        shutil.rmtree(test_memory_dir)
    test_memory_dir.mkdir(parents=True)
    
    # Mock paths to point to test_memory_dir
    # We can patch MEMORY_PATH in store.py or just rely on the main MEMORY_DIR if we are careful.
    # The safest is to patch src.memory.store.MEMORY_PATH
    
    with patch("src.memory.store.MEMORY_PATH", test_memory_dir):
        
        # Initial State
        state = AppState(
            selected_paper={"title": "Test Paper", "summary": "Summary"},
            trending_keywords=["AI"],
            post_draft="Draft 1",
            memory={},
            memory_events=[]
        )
        
        # --- Step 2: Approval (User Edits Manual) ---
        # User manual edit
        with patch("src.agents.human_approval.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "type": "edit",
                "args": {"instruction": "Fix typo", "draft": "Draft 2 (Manual)"}
            }
            res_step2 = await human_approval(state)
            
            # Apply updates to state
            state.post_draft = res_step2["post_draft"]
            state.revision_requested = res_step2["revision_requested"]
            state.edit_requests = res_step2["edit_requests"]
            state.revision_history = res_step2["revision_history"]
            state.memory_events = res_step2["memory_events"]
            state.post_history = res_step2["post_history"]
            
            assert state.post_draft == "Draft 2 (Manual)"
            assert len(state.edit_requests) == 1
            assert state.edit_requests[0]["type"] == "edit"

        # --- Step 4: Approval (User requests revision via chat/response) ---
        # Current logic: Response -> Revision Request
        with patch("src.agents.human_approval.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {
                "type": "response",
                "args": "Make it shorter"
            }
            res_step4 = await human_approval(state)
            
            state.approved = res_step4["approved"]
            state.revision_requested = res_step4["revision_requested"]
            state.edit_requests = res_step4["edit_requests"]
            state.memory_events = res_step4["memory_events"]
            
            assert state.revision_requested is True
            assert len(state.edit_requests) == 2
            assert state.edit_requests[1]["instruction"] == "Make it shorter"
            assert state.edit_requests[1]["type"] == "response"

        # --- Step 5: Writer generates Draft 3 ---
        class MockLLM:
            def bind_tools(self, tools):
                return self
            async def ainvoke(self, input_val, **kwargs):
                # input_val is messages (list) or similar
                return MagicMock(content="Draft 3 (Short)", tool_calls=[])

        with patch("src.agents.post_writer.settings") as mock_settings, \
             patch("src.agents.post_writer.init_chat_model") as mock_init_llm:
            
            mock_settings.openai_api_key = "fake_key"
            mock_settings.tavily_api_key = "fake" # Enable tools path
            
            mock_init_llm.return_value = MockLLM()
            
            res_step5 = await write_post(state)
            
            state.post_draft = res_step5["post_draft"]
            state.post_history = res_step5["post_history"]
            
            assert state.post_draft == "Draft 3 (Short)"
            assert len(state.post_history) > 0

        # --- Step 6: Approval (Accept) ---
        with patch("src.agents.human_approval.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"type": "accept", "args": "Great job"}
            res_step6 = await human_approval(state)
            
            state.approved = res_step6["approved"]
            state.memory_events = res_step6["memory_events"]
            
            assert state.approved is True
            # Should have events from Step 2, Step 4, Step 6
            # Step 2: edit (Kind: post_style_feedback)
            # Step 4: response (Kind: post_style_feedback)
            # Step 6: accept (Kind: post_style_feedback)
            assert len(state.memory_events) == 3

        # --- Step 7: Memory Updater ---
        # Mock LLMs for memory extraction
        with patch("src.agents.memory_updater.init_chat_model") as mock_init_mem:
            mock_llm_mem = MagicMock()
            mock_init_mem.return_value = mock_llm_mem
            
            # Mock with_structured_output
            mock_struct = MagicMock()
            mock_llm_mem.with_structured_output.return_value = mock_struct
            
            # The chain.ainvoke calls
            mock_struct.ainvoke.return_value = MagicMock(model_dump=lambda: {})
            # We assume it "works" and returns empty updates, effectively just checking the flow doesn't crash
            # and that memory_events are cleared.
            
            res_step7 = await update_memory(state)
            
            assert len(res_step7["memory_events"]) == 0
            
            # Verify events were processed? 
            # We can verify save() was called if we mocked MemoryStore details, 
            # but here we are measuring end-to-end "it runs and clears events".
            # Real file persistence:
            assert (test_memory_dir / "topic_preferences.json").exists()
            
    # Cleanup
    if test_memory_dir.exists():
        shutil.rmtree(test_memory_dir)
