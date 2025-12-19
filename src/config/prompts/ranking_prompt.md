Rank these papers by relevance to the topic "{topic}", user interests "{interests}", and the latest conversation context "{conversation}".
If the user explicitly mentions a preference or a candidate (by title or index) in the conversation, prioritize that.
Papers: {papers}

Return an object with:
- index: the 0-based index of the best paper (e.g., 0 for the first paper)
- rationale: (optional) short reason for the choice
