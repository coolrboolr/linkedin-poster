# LinkedIn Poster Agent

A LangGraph-based agent that scans Google Trends, fetches relevant ArXiv papers, and drafts LinkedIn posts tailored to your preferences.

## Features

- **Trend Scanning**: Monitors Google Trends for ML/AI topics.
- **ArXiv Integration**: Fetches latest papers matching trending keywords.
- **Relevance Ranking**: Scores papers based on your interests and trends.
- **Conversational Agent**: Asks clarifying questions to refine post angle.
- **Revision-Aware Drafting**: Keeps chat history, revision history, and prior drafts so each rewrite builds on every edit/instruction.
- **Memory**: Remembers your topic preferences, comprehension style, and post formatting.
- **Human-in-the-Loop**: Review, edit, and approve posts before they are finalized.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install .
    ```

2.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in your API keys.
    ```bash
    cp .env.example .env
    ```

3.  **Run the Agent (blocking-friendly)**:
    ```bash
    langgraph dev --allow-blocking
    ```
    The `--allow-blocking` flag enables the interrupt-driven nodes (conversation + approvals) to pause and resume correctly in the dev server. For direct execution you can still run:
    ```bash
    python src/graph.py
    ```

4.  **Use Agent Inbox for interrupts**:
    - Start the dev server as above, then open Agent Inbox (LangGraph Studio) in your browser.
    - Each interrupting node (conversation and human approval) will surface a prompt in the Inbox with Accept / Respond / Edit / Ignore actions.
    - The graph will pause until you choose an action; your response is fed back into the graph state to continue execution.

## Testing

Run tests with LangSmith tracing enabled:

```bash
dotenv run pytest
```

## Architecture

The project uses a multi-agent architecture orchestrated by LangGraph:

1.  **Trend Scanner**: Finds what's hot.
2.  **ArXiv Fetcher**: Gets the science.
3.  **Relevance Ranker**: Filters for you.
4.  **Conversation**: Refines the output.
5.  **Post Writer**: Drafts the content.
6.  **Human Approval**: You have the final say.
7.  **Memory Updater**: Learns from your feedback.
