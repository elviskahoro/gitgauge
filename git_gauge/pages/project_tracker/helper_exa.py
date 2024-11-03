from __future__ import annotations

from exa_py import Exa

from git_gauge.helper_exa.helper_exa import exa_api_call


def get_swot_from_exa_strenghts(
    organization_url: str,
    client: Exa | None = None,
) -> str:
    if client is None:
        raise ValueError("Exa Client is required")

    prompt: str = f"What are the the strengths of the organization: {organization_url}?"
    exa_summary = exa_api_call(
        prompt=prompt,
        client=client,
    )
    return exa_summary


def get_swot_from_exa_weaknesses(
    organization_url: str,
    client: Exa | None = None,
) -> str:
    if client is None:
        raise ValueError("Exa Client is required")

    prompt: str = (
        f"What are the the weaknesses of the organization: {organization_url}?"
    )
    exa_summary = exa_api_call(
        prompt=prompt,
        client=client,
    )
    return exa_summary
