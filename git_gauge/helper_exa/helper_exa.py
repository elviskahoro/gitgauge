from __future__ import annotations

from exa_py import Exa


def set_up_client_from_tokens(
    tokens: list[str],
) -> Exa:
    return Exa(
        api_key=tokens.get("EXA_API_KEY"),
    )

def exa_api_call(
    prompt: str,
    client: Exa,
) -> str:
    """Make an API call to Exa to search and return content summaries."""
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
    return exa_summary

