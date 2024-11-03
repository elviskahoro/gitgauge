"""Models related to hackathon projects."""
# trunk-ignore-all(pyright/reportMissingTypeStubs)
from __future__ import annotations

from .base import Base


class Swot(Base, table=True): # trunk-ignore(pyright/reportGeneralTypeIssues,pyright/reportCallIssue)

    repo_path: str
    strength: str
    weakness: str

    def __hash__(
        self: Swot,
    ) -> int:
        """The hash is based on the repository path, ensuring that each
        project can be uniquely identified by its repository path.

        Returns:
            int: The github repo path as an integer hash value.
        """
        return hash(self.repo_path)
