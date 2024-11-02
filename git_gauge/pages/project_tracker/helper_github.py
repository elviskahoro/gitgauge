from __future__ import annotations

from typing import TYPE_CHECKING, Generator

from github import Github
from github.GithubException import GithubException

from git_gauge import helper_utils
from git_gauge.helper_logging import Severity, log
from git_gauge.otel import tracer

if TYPE_CHECKING:
    from github.PullRequest import PullRequest
    from github.Repository import Repository


def set_up_client_from_tokens(
    tokens: dict[
        str,
        str | None,
    ],
) -> Github:
    error_message: str | None = None
    tokens = helper_utils.check_tokens(
        tokens=tokens,
    )
    log(
        the_log="Setting up client",
        severity=Severity.DEBUG,
        file=__file__,
    )
    github_owner: str | None = tokens.get(
        "GITHUB_OWNER",
        None,
    )
    if github_owner is None:
        error_message = "GITHUB_OWNER is not set"
        raise AttributeError(error_message)

    github_repo: str | None = tokens.get(
        "GITHUB_REPO",
        None,
    )
    if github_repo is None:
        error_message = "GITHUB_REPO is not set"
        raise AttributeError(error_message)

    github_client_id: str | None = tokens.get(
        "GITHUB_CLIENT_ID",
        None,
    )
    if github_client_id is None:
        error_message = "GITHUB_CLIENT_ID is not set"
        raise AttributeError(error_message)

    github_client_secret: str | None = tokens.get(
        "GITHUB_CLIENT_SECRET",
        None,
    )
    if github_client_secret is None:
        error_message = "GITHUB_CLIENT_SECRET is not set"
        raise AttributeError(error_message)

    github_client: Github = Github()
    github_client.get_oauth_application(
        client_id=github_client_id,
        client_secret=github_client_secret,
    )
    return github_client


def extract_repo_path_from_url(
    url: str,
) -> str:
    """Extract repository path from GitHub URL.

    Args:
        url: GitHub repository URL (e.g., 'https://github.com/owner/repo' or 'github.com/owner/repo')

    Returns:
        Repository path in format 'owner/repo'

    Raises:
        ValueError: If URL format is invalid
    """
    # Remove protocol prefix if present
    url = url.replace("https://", "").replace("http://", "")

    # Remove github.com prefix if present
    if url.startswith("github.com/"):
        url = url.replace("github.com/", "")

    # Remove trailing slash and .git suffix if present
    url = url.rstrip("/").replace(".git", "")

    # Validate format (should be owner/repo)
    parts = url.split("/")
    if len(parts) != 2:
        raise ValueError("Invalid GitHub URL format. Expected format: owner/repo")

    return url


def fetch_repo(
    repo_path: str,
    client: Github | None,
) -> Repository | None:
    with tracer.start_as_current_span("fetch_repo") as span:
        if client is None:
            span.add_event(
                name="fetch_repo-no_client",
                attributes={
                    "repo_path": repo_path,
                },
            )
            raise AttributeError("No client provided")

        span.add_event(
            name="fetch_repo-started",
            attributes={
                "repo_path": repo_path,
            },
        )
        repo: Repository | None = None
        try:
            repo = client.get_repo(repo_path)
            span.add_event(
                name="fetch_repo-completed",
                attributes={
                    "repo_path": repo_path,
                    "repo_full_name": repo.full_name,
                },
            )

        except GithubException as e:
            span.record_exception(e)
            span.add_event(
                name="fetch_repo-error",
                attributes={
                    "repo_path": repo_path,
                },
            )

        return repo


def fetch_pull_requests(
    repo: Repository,
) -> Generator[PullRequest, None, None]:
    span_name: str = "fetch_pull_requests"
    with tracer.start_as_current_span(span_name) as span:
        span.add_event(
            name=f"{span_name}-started",
        )
        yield from repo.get_pulls(
            state="all",
            sort="updated",
            direction="desc",
        )
        span.add_event(
            name=f"{span_name}-completed",
        )


def fetch_pull_request_for_repo(
    repo: Repository,
) -> Generator[PullRequest, None, None]:
    span_name: str = "fetch_pull_request_for_repo"
    with tracer.start_as_current_span(span_name) as span:
        span.add_event(
            name=f"{span_name}-started",
        )
        yield from fetch_pull_requests(
            repo=repo,
        )
        span.add_event(
            name=f"{span_name}-completed",
        )
