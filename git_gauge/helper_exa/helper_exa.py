from __future__ import annotations

from exa_py import Exa


def exa_api_call(
        prompt: str,
        client: Exa | None = None,
    ):
    """Make an API call to Exa to search and return content summaries."""
    try:
        # Perform the search with the given prompt and filters
        exa_summary = client.search_and_contents(
            query=prompt,
            type="neural",
            use_autoprompt=True,
            num_results=20,
            text=True,
            exclude_domains=["en.wikipedia.org"],
            start_published_date="2023-01-01",
            category="tweet",
        )

    except Exception as e:
        print(f"Error: {e}")
        return "Failed to generate summary due to an error."

    return exa_summary
