from src.agents.post_writer import write_post
from src.config.settings import settings
from src.state import AppState
from unittest.mock import MagicMock, patch
import pytest

@pytest.mark.asyncio
async def test_post_writer_basic():
    paper = {"title": "Test Paper", "summary": "Summary"}
    state = AppState(selected_paper=paper)
    
    # Mock ChatOpenAI and its invoke method
    with patch('src.agents.post_writer.init_chat_model') as MockInitModel, \
         patch('src.agents.post_writer.PROMPTS_DIR') as MockPromptsDir:
        
        # Mock prompt file existence and read
        mock_prompt_file = MagicMock()
        mock_prompt_file.exists.return_value = True
        mock_prompt_file.read_text.return_value = "Prompt Template"
        MockPromptsDir.__truediv__.return_value = mock_prompt_file
        
        mock_llm = MockInitModel.return_value
        mock_runnable = MagicMock()
        # Make ainvoke an async method
        async def async_return(*args, **kwargs):
            mock_content = MagicMock()
            mock_content.content = "Generated Draft Post"
            return mock_content
        mock_runnable.ainvoke.side_effect = async_return
        
        with patch('src.agents.post_writer.ChatPromptTemplate') as MockPrompt:
             mock_template = MockPrompt.from_template.return_value
             mock_template.__or__.return_value = mock_runnable
             
             updates = await write_post(state)
            
             assert "post_draft" in updates
             assert updates["post_draft"] == "Generated Draft Post"


@pytest.mark.asyncio
async def test_post_writer_respects_format_preferences():
    paper = {"title": "Test Paper", "summary": "Summary"}
    state = AppState(
        selected_paper=paper,
        memory={
            "post_format_preferences": {
                "length": "short",
                "emojis": True,
                "hashtags": True,
                "max_iterations": 1,
                "cta": "Invite readers to share their takeaways",
            }
        },
    )

    with patch('src.agents.post_writer.init_chat_model') as MockInitModel, \
         patch('src.agents.post_writer.PROMPTS_DIR') as MockPromptsDir, \
         patch('src.agents.post_writer.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.post_writer.settings') as mock_settings:

        mock_settings.openai_api_key = "test-key"
        mock_settings.llm_model = "test-model"

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template: {format}"
        MockPromptsDir.__truediv__.return_value = mock_prompt_file

        mock_llm = MockInitModel.return_value
        mock_runnable = MagicMock()

        async def async_return(inputs, *args, **kwargs):
            fmt = inputs["format"]
            assert "Length: short." in fmt
            assert "Use emojis." in fmt
            assert "Include relevant hashtags." in fmt
            assert "Generate 1 variation(s) only." in fmt
            assert "cta: Invite readers to share their takeaways" in fmt
            result = MagicMock()
            result.content = "Generated Draft Post"
            return result

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        updates = await write_post(state)
        assert updates["post_draft"] == "Generated Draft Post"


@pytest.mark.asyncio
async def test_post_writer_no_paper():
    state = AppState(selected_paper=None)
    updates = await write_post(state)
    assert updates["post_draft"] == "Error: No paper selected."


@pytest.mark.asyncio
async def test_post_writer_missing_prompt_file():
    paper = {"title": "Test", "summary": "Summary"}
    state = AppState(selected_paper=paper)

    with patch('src.agents.post_writer.PROMPTS_DIR') as MockPromptsDir:
        mock_path = MagicMock()
        mock_path.read_text.side_effect = FileNotFoundError()
        MockPromptsDir.__truediv__.return_value = mock_path

        updates = await write_post(state)
        assert updates["post_draft"] == "Error: Prompt missing."


@pytest.mark.asyncio
async def test_post_writer_missing_api_key():
    paper = {"title": "Test", "summary": "Summary"}
    state = AppState(selected_paper=paper)

    with patch('src.agents.post_writer.PROMPTS_DIR') as MockPromptsDir, \
         patch('src.agents.post_writer.settings') as mock_settings:

        mock_settings.openai_api_key = None
        mock_path = MagicMock()
        mock_path.read_text.return_value = "Template"
        MockPromptsDir.__truediv__.return_value = mock_path

        updates = await write_post(state)
        assert updates["post_draft"] == "Error: API Key missing."


@pytest.mark.asyncio
async def test_post_writer_includes_all_edit_requests():
    paper = {"title": "Test Paper", "summary": "Summary"}
    state = AppState(
        selected_paper=paper,
        edit_requests=[
            {"instruction": "Shorten the intro", "source": "human_approval", "type": "response"},
            {"instruction": "Add a CTA at the end", "source": "human_approval", "type": "edit"},
        ],
    )

    with patch('src.agents.post_writer.init_chat_model') as MockInitModel, \
         patch('src.agents.post_writer.PROMPTS_DIR') as MockPromptsDir, \
         patch('src.agents.post_writer.ChatPromptTemplate') as MockPrompt, \
         patch('src.agents.post_writer.settings') as mock_settings:

        mock_settings.openai_api_key = "test-key"
        mock_settings.llm_model = "test-model"

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template: {all_edit_requests}"
        MockPromptsDir.__truediv__.return_value = mock_prompt_file

        mock_llm = MockInitModel.return_value
        mock_runnable = MagicMock()

        async def async_return(inputs, *args, **kwargs):
            edits = inputs["all_edit_requests"]
            assert "Shorten the intro" in edits
            assert "Add a CTA at the end" in edits
            result = MagicMock()
            result.content = "Generated Draft Post"
            return result

        mock_runnable.ainvoke.side_effect = async_return

        mock_template = MockPrompt.from_template.return_value
        mock_template.__or__.return_value = mock_runnable

        updates = await write_post(state)
        assert updates["post_draft"] == "Generated Draft Post"


@pytest.mark.asyncio
async def test_post_writer_uses_tools_when_available(monkeypatch):
    paper = {"title": "Test Tool Paper", "summary": "Summary"}
    state = AppState(selected_paper=paper, memory={"post_format_preferences": {}})

    tool_calls = {}

    class DummyTool:
        name = "search_web"

        def invoke(self, args):
            tool_calls["args"] = args
            return "tool result"

        async def ainvoke(self, args, config=None, **kwargs):
            tool_calls["args"] = args
            return "tool result"

    class LLMStub:
        def __init__(self):
            self.calls = 0
            self.bound_tools = []

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Resp",
                    (),
                    {
                        "content": "draft with tools",
                        "tool_calls": [
                            {"name": self.bound_tools[0].name, "id": "call-1", "args": {"query": "ai"}}  # type: ignore[index]
                        ],
                    },
                )()
            return type("Resp", (), {"content": "Final Draft", "tool_calls": []})()

    class PromptStub:
        def __init__(self, text):
            self.text = text

        def format_messages(self, **_):
            return ["formatted"]

    with patch.object(settings, "openai_api_key", "key", create=True), \
         patch.object(settings, "tavily_api_key", "tv", create=True), \
         patch.object(settings, "llm_model", "gpt-stub", create=True), \
         patch("src.agents.post_writer.init_chat_model", return_value=LLMStub()), \
         patch("src.agents.post_writer.ChatPromptTemplate.from_template", return_value=PromptStub("t")), \
         patch("src.agents.post_writer.PROMPTS_DIR") as mock_prompts_dir:

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template"
        mock_prompts_dir.__truediv__.return_value = mock_prompt_file

        tool = DummyTool()
        monkeypatch.setattr("src.agents.post_writer.TOOLS", [tool])
        monkeypatch.setattr("src.agents.post_writer.TOOL_MAP", {"search_web": tool})

        updates = await write_post(state)

    assert updates["post_draft"] == "Final Draft"
    assert tool_calls["args"] == {"query": "ai"}


@pytest.mark.asyncio
async def test_post_writer_handles_tool_failure(monkeypatch):
    paper = {"title": "Test Tool Paper", "summary": "Summary"}
    state = AppState(selected_paper=paper, memory={"post_format_preferences": {}})

    class FailingTool:
        name = "search_web"

        def invoke(self, _args):
            raise RuntimeError("boom")

        async def ainvoke(self, _args, config=None, **kwargs):
            raise RuntimeError("boom")

    class LLMStub:
        def __init__(self):
            self.calls = 0
            self.bound_tools = []

        def bind_tools(self, tools):
            self.bound_tools = tools
            return self

        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return type(
                    "Resp",
                    (),
                    {
                        "content": "draft with tools",
                        "tool_calls": [
                            {"name": self.bound_tools[0].name, "id": "call-1", "args": {"query": "ai"}}  # type: ignore[index]
                        ],
                    },
                )()
            return type("Resp", (), {"content": "Final Draft", "tool_calls": []})()

    class PromptStub:
        def __init__(self, text):
            self.text = text

        def format_messages(self, **_):
            return ["formatted"]

    with patch.object(settings, "openai_api_key", "key", create=True), \
         patch.object(settings, "tavily_api_key", "tv", create=True), \
         patch.object(settings, "llm_model", "gpt-stub", create=True), \
         patch("src.agents.post_writer.init_chat_model", return_value=LLMStub()), \
         patch("src.agents.post_writer.ChatPromptTemplate.from_template", return_value=PromptStub("t")), \
         patch("src.agents.post_writer.PROMPTS_DIR") as mock_prompts_dir:

        mock_prompt_file = MagicMock()
        mock_prompt_file.read_text.return_value = "Template"
        mock_prompts_dir.__truediv__.return_value = mock_prompt_file

        tool = FailingTool()
        monkeypatch.setattr("src.agents.post_writer.TOOLS", [tool])
        monkeypatch.setattr("src.agents.post_writer.TOOL_MAP", {"search_web": tool})

        updates = await write_post(state)

    assert updates["post_draft"] == "Final Draft"
