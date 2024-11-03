from __future__ import annotations

from git_gauge.helper_perplexity import (
    DEFAULT_PROMPT,
    ChatCompletion,
    ChatResponse,
    Client,
    Message,
)


STRENGTH_PROMPT: str = """
Give a one line summary of the strengths of the product or service of this website <link_to_website>.
"""

WEAKNESS_PROMPT: str = """
Give a one line summary of the weaknesses of the product or service of this website <link_to_website>.
"""


async def perplexity_get_product_or_service(
    website_url: str,
    strength: bool = False,
    client: Client | None = None,
) -> str:
    if client is None:
        raise AssertionError("Client is required")

    if strength:
        content: str = STRENGTH_PROMPT.replace(
            "<link_to_website>",
            website_url,
        )
    else:
        content: str = WEAKNESS_PROMPT.replace(
            "<link_to_website>",
            website_url,
        )

    chat_completion: ChatCompletion = ChatCompletion(
        messages=[
            Message(
                content=content,
            ),
        ],
    )
    chat_response: ChatResponse = await client.call_perplexity_api(
        chat_completion=chat_completion,
    )
    return chat_response.get_content()
