# LinkedIn Poster Agent

A LangGraph-based agent that scans Google Trends, fetches relevant ArXiv papers, and drafts LinkedIn posts tailored to your preferences.

## Features

- **Trend Scanning**: Monitors Google Trends for ML/AI topics.
- **ArXiv Integration**: Fetches latest papers matching trending keywords.
- **Relevance Ranking**: Scores papers based on your interests and trends.
- **Conversational Agent**: Asks clarifying questions to refine post angle.
- **Memory**: Remembers your topic preferences, comprehension style, and post formatting.
- **Human-in-the-Loop**: Review and approve posts before they are finalized.

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

3.  **Run the Agent**:
    ```bash
    python src/graph.py
    ```

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
