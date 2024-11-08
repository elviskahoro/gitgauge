from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Generator

from github import Github
from github.GithubException import GithubException

from git_gauge import helper_utils
from git_gauge.helper_logging import Severity, log
from git_gauge.otel import tracer

if TYPE_CHECKING:
    from github.Issue import Issue
    from github.PullRequest import PullRequest
    from github.Repository import Repository


class Since(Enum):
    """Enum for the since parameter of the GitHub API."""

    WEEK = "week"
    MONTH = "month"

    def get_datetime(
        self: Since,
    ) -> datetime:
        match self:
            case Since.WEEK:
                return datetime.now(
                    tz=datetime.timezone.utc,
                ) - timedelta(
                    days=7,
                )

            case Since.MONTH:
                return datetime.now(
                    tz=datetime.timezone.utc,
                ) - timedelta(
                    days=30,
                )

        error_message: str = f"Invalid since value: {self}"
        raise ValueError(error_message)


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


def extract_repo_path(
    repo_search_input: str,
) -> str:
    """Extract repository path from GitHub URL.

    Args:
        repo_search_input: GitHub repository URL (e.g., 'https://github.com/owner/repo' or 'github.com/owner/repo')

    Returns:
        Repository path in format 'owner/repo'

    Raises:
        ValueError: If URL format is invalid
    """
    # Remove protocol prefix if present
    repo_search_input = repo_search_input.replace("https://", "").replace("http://", "")

    # Remove github.com prefix if present
    if repo_search_input.startswith("github.com/"):
        repo_search_input = repo_search_input.replace("github.com/", "")

    # Remove trailing slash and .git suffix if present
    repo_search_input = repo_search_input.rstrip("/").replace(".git", "")

    # Validate format (should be owner/repo)
    parts = repo_search_input.split("/")
    if len(parts) != 2:
        raise ValueError("Invalid GitHub URL format. Expected format: owner/repo")

    return repo_search_input


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
            raise AttributeError("No GitHub client provided")

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


def fetch_pull_requests_for_repo(
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


def _get_since_datetime(
    since_enum: Since | None,
    since_datetime: datetime | None,
) -> datetime:
    error_message: str | None = None
    match since_enum, since_datetime:
        case None, None:
            error_message = "Either since_enum or since_datetime must be provided"
            raise ValueError(error_message)

        case _, None:
            return since_enum.get_datetime()

        case None, _:
            return since_datetime

        case _, _:
            error_message = "Both since_enum and since_datetime cannot be provided"
            raise ValueError(error_message)

def fetch_github_issues_for_repo(
    repo: Repository,
    since_enum: Since | None = None,
    since_datetime: datetime | None = None,
) -> Generator[Issue, None, None]:
    span_name: str = "fetch_github_issues_for_repo"
    with tracer.start_as_current_span(span_name) as span:
        span.add_event(
            name=f"{span_name}-started",
        )
        since: datetime = _get_since_datetime(
            since_enum=since_enum,
            since_datetime=since_datetime,
        )
        yield from repo.get_issues(
            state="all",
            since=since,
        )
