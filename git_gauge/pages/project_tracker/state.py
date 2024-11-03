# trunk-ignore-all(ruff/PLW0603)
from __future__ import annotations

from typing import TYPE_CHECKING, Generator

import reflex as rx
from exa_py import Exa
from github import Github as ClientGithub
from sqlalchemy import select
from sqlalchemy.orm import Session

from git_gauge.models.swot import Swot

from chromadb import HttpClient as ClientChroma
from git_gauge.helper_perplexity import Client as ClientPerplexity

from git_gauge import helper_chroma, helper_perplexity, helper_exa
from git_gauge.models.project import Project
from git_gauge.otel import tracer
from git_gauge.pages.project_tracker import helper_github
from git_gauge.tokens import TOKENS

from .constants import (
    DEFAULT_DISTANCE_THRESHOLD_FOR_VECTOR_SEARCH,
    NUMBER_OF_RESULTS_TO_DISPLAY_FOR_VECTOR_SEARCH,
    NUMBER_OF_WORDS_TO_DISPLAY_FOR_REPO_DESCRIPTION,
)
from .helper_chroma import chroma_add_project, chroma_get_projects
from .helper_exa import get_swot_from_exa_strenghts, get_swot_from_exa_weaknesses
from .repo_cards import (
    repo_card_description_component,
    repo_card_skeleton,
    repo_card_stats_component,
)

if TYPE_CHECKING:
    from github.Repository import Repository

CLIENT_CHROMA: ClientChroma | None = None
CLIENT_GITHUB: ClientGithub | None = None
CLIENT_PERPLEXITY: ClientPerplexity | None = None
CLIENT_EXA: Exa | None = None


def set_up_clients() -> None:
    global CLIENT_CHROMA, CLIENT_GITHUB, CLIENT_PERPLEXITY, CLIENT_EXA
    with tracer.start_as_current_span("set_up_clients") as span:
        try:
            span.add_event(
                name="set_up_clients-chroma_client-started",
            )
            CLIENT_CHROMA = helper_chroma.set_up_client_from_tokens(
                tokens=TOKENS,
            )

        except AttributeError:
            span.add_event(
                name="set_up_clients-chroma_client-failed",
            )

        try:
            span.add_event(
                name="set_up_clients-github_client-started",
            )
            CLIENT_GITHUB = helper_github.set_up_client_from_tokens(
                tokens=TOKENS,
            )

        except AttributeError:
            span.add_event(
                name="set_up_clients-github_client-failed",
            )

        try:
            span.add_event(
                name="set_up_clients-perplexity_client-started",
            )
            CLIENT_PERPLEXITY = helper_perplexity.Client.set_up_client_from_tokens(
                tokens=TOKENS,
            )

        except AttributeError:
            span.add_event(
                name="set_up_clients-perplexity_client-failed",
            )

        try:
            span.add_event(
                name="set_up_clients-exa_client-started",
            )
            CLIENT_EXA = helper_exa.set_up_client_from_tokens(
                tokens=TOKENS,
            )

        except AttributeError:
            span.add_event(
                name="set_up_clients-exa_client-failed",
            )


set_up_clients()


class State(rx.State):
    """The state for the project tracker page."""

    distance_threshold: int = DEFAULT_DISTANCE_THRESHOLD_FOR_VECTOR_SEARCH
    has_generated_audio: bool = False
    current_filter_vector_search_text: str = ""
    last_vector_search_filter_text: str = ""
    repo_path_search: str = ""
    ag_grid_selection_repo_path: str | None = None
    ag_grid_selection_project_index: int | None = None

    projects: list[Project] = []
    projects_to_commit: list[Project] = []
    display_data_indices: list[int] = []

    def distance_threshold_setter(
        self,
        value: list[int],
    ) -> None:
        self.distance_threshold = value[0]

    def distance_threshold_commit(
        self,
        value: list[int],
    ) -> None:
        del value
        self.vector_search_filter()

    def get_swot(
        self: State,
        repo_path: str,
    ) -> Swot | None:
        with rx.session() as session:
            swot: Swot | None = (
                session.query(Swot).filter(Swot.repo_path == repo_path).first()
            )
            return swot

    @staticmethod
    def _find_project_index_using_repo_path(
        projects: list[Project],
        repo_path: str | None,
    ) -> Project | None:
        if repo_path is None:
            return None

        first_index = next(
            (i for i in range(len(projects)) if projects[i].repo_path == repo_path),
            None,
        )
        if first_index is None:
            return None

        return first_index

    @rx.var(cache=True)
    def has_selected_data(
        self,
    ) -> bool:
        return self.ag_grid_selection_repo_path is not None

    @rx.var(cache=True)
    def display_data(
        self,
    ) -> list[dict]:
        return [self.projects[i].to_ag_grid_dict() for i in self.display_data_indices]

    @rx.var
    def repo_card_stats(
        self,
    ) -> rx.Component:
        project_index: int | None = State._find_project_index_using_repo_path(
            projects=self.projects,
            repo_path=self.ag_grid_selection_repo_path,
        )
        if project_index is None:
            return rx.fragment(repo_card_skeleton())

        project: Project = self.projects[project_index]
        return rx.fragment(
            repo_card_stats_component(
                repo_path=project.repo_path,
                repo_url=f"https://github.com/{project.repo_path}",
                stars=f"{project.stars:,}",
                language=f"{project.language}",
                website_url=f"{project.website}",
            ),
        )

    @rx.var
    def repo_card_description(
        self,  # trunk-ignore(ruff/ANN10)
    ) -> rx.Component:
        project_index: int | None = State._find_project_index_using_repo_path(
            projects=self.projects,
            repo_path=self.ag_grid_selection_repo_path,
        )
        if project_index is None:
            return rx.fragment(
                repo_card_skeleton(),
            )

        project: Project = self.projects[project_index]
        description: str = str(project.description)
        if first_n_words_from_description := " ".join(
            project.description.split()[
                :NUMBER_OF_WORDS_TO_DISPLAY_FOR_REPO_DESCRIPTION
            ],
        ):
            description = first_n_words_from_description

        return rx.fragment(
            repo_card_description_component(
                description=description,
            ),
        )

    @rx.var
    def repo_card_strength(
        self,
    ) -> rx.Component:
        repo_path: str | None = self.ag_grid_selection_repo_path
        if repo_path is None:
            return repo_card_skeleton()

        strength_swot: Swot | None = self.get_swot(
            repo_path=repo_path,
        )
        if strength_swot is None:
            return repo_card_skeleton()

        return rx.card(
            rx.vstack(
                rx.heading("Strength"),
                rx.text(strength_swot.strengths),
            ),
            size="5",
        )

    @rx.var
    def repo_card_weakness(
        self,
    ) -> rx.Component:
        repo_path: str | None = self.ag_grid_selection_repo_path
        if repo_path is None:
            return repo_card_skeleton()

        weakness_swot: Swot | None = self.get_swot(
            repo_path=repo_path,
        )
        if weakness_swot is None:
            return repo_card_skeleton()

        return rx.card(
            rx.vstack(
                rx.heading("Weakness"),
                rx.text(weakness_swot.weaknesses),
            ),
            size="5",
        )

    def setter_repo_path_search(
        self: State,
        repo_path: str,
    ) -> None:
        with tracer.start_as_current_span("setter_repo_path_search") as span:
            self.repo_path_search = repo_path
            span.add_event(
                name="repo_path_current-set",
                attributes={
                    "repo_path": str(self.repo_path_search),
                },
            )

    def clear_repo_path_search(
        self: State,
    ) -> None:
        with tracer.start_as_current_span("clear_repo_path_search") as span:
            self.repo_path_search = ""
            span.add_event(
                name="repo_path_current-clear",
            )

    def setter_repo_path_ag_grid_selection(
        self: State,
        rows: list[dict[str, str]],
        _0: int,
        _1: int,
    ) -> None:
        del _0, _1
        with tracer.start_as_current_span("setter_repo_path_ag_grid_selection") as span:
            if rows and (repo_path := rows[0].get("repo_path")):
                self.ag_grid_selection_repo_path = repo_path
                span.add_event(
                    name="selection_repo_path-set",
                    attributes={
                        "repo_path": str(self.ag_grid_selection_repo_path),
                    },
                )
                return

            span.add_event(
                name="selection_repo_path-unset",
            )

    def setter_repo_filter_vector_search_text(
        self: State,
        repo_filter_vector_search_text: str,
    ) -> None:
        with tracer.start_as_current_span(
            "setter_repo_filter_vector_search_text",
        ) as span:
            self.current_filter_vector_search_text = repo_filter_vector_search_text
            span.add_event(
                name="repo_filter_vector_search_text-set",
                attributes={
                    "repo_filter_vector_search_text": str(
                        self.current_filter_vector_search_text,
                    ),
                },
            )
            if not repo_filter_vector_search_text:
                return

            self.last_vector_search_filter_text = repo_filter_vector_search_text

    def display_data_indices_setter(
        self: State,
        display_data_indices: list[int],
    ) -> None:
        with tracer.start_as_current_span("display_data_indices_setter") as span:
            self.display_data_indices = display_data_indices
            span.add_event(
                name="display_data_indices-set",
                attributes={
                    "display_data_indices": str(self.display_data_indices),
                },
            )

    def add_project_to_display_data(
        self: State,
        project: Project,
    ) -> Generator[None, None, None]:
        with tracer.start_as_current_span("add_project_to_display_data") as span:

            def add_project() -> int:
                span.add_event(
                    name="project-add_project-started",
                    attributes={
                        "project_repo_path": str(project.repo_path),
                    },
                )
                self.projects.append(project)
                project_index = self._find_project_index_using_repo_path(
                    projects=self.projects,
                    repo_path=project.repo_path,
                )
                if project_index is None:
                    error_msg: str = (
                        f"Project should be non null and found in projects: {project.repo_path}"
                    )
                    raise AssertionError(error_msg)

                return project_index

            project_index: int | None = self._find_project_index_using_repo_path(
                projects=self.projects,
                repo_path=project.repo_path,
            )
            span.add_event(
                name="project-find_index-completed",
                attributes={
                    "project_repo_path": str(project.repo_path),
                    "project_index": str(project_index),
                },
            )
            if project_index is None:
                span.add_event(
                    name="project-add_project-queued",
                    attributes={
                        "project_repo_path": str(project.repo_path),
                    },
                )
                project_index = add_project()
                span.add_event(
                    name="project-add_project-completed",
                    attributes={
                        "project_repo_path": str(project.repo_path),
                    },
                )

            display_data_indices: list[int] = [*self.display_data_indices]
            span.add_event(
                name="display_data_indices-current",
                attributes={
                    "display_data_indices": str(display_data_indices),
                },
            )
            if project_index in display_data_indices:
                span.add_event(
                    name="project-already_in_display_data",
                    attributes={
                        "project_repo_path": str(project.repo_path),
                    },
                )

            if project_index not in display_data_indices:
                display_data_indices.append(project_index)
                self.display_data_indices_setter(
                    display_data_indices=display_data_indices,
                )
                span.add_event(
                    name="project-added_to_display_data",
                    attributes={
                        "project_repo_path": str(project.repo_path),
                    },
                )

    def _save_projects_to_db(
        self: State,
        projects: list[Project],
    ) -> Generator[None, None, None]:
        def filter_projects_to_save(
            db: Session,
            projects: list[Project],
        ) -> list[Project]:
            with tracer.start_as_current_span("get_new_projects") as span:
                span.add_event(
                    name="get_new_projects-started",
                    attributes={
                        "project_count-to_commit": len(projects),
                    },
                )
                existing_repo_paths: set[str] = {
                    str(path[0]) for path in db.query(Project.repo_path).all()
                }
                span.add_event(
                    name="get_new_projects-existing_repo_paths",
                    attributes={
                        "existing_repo_paths": str(existing_repo_paths),
                        "existing_repo_path_count": len(existing_repo_paths),
                        "existing_repo_path_type": str(type(existing_repo_paths)),
                    },
                )
                new_projects: list[Project] = [
                    project
                    for project in projects
                    if str(project.repo_path) not in existing_repo_paths
                ]
                span.add_event(
                    name="get_new_projects-completed",
                    attributes={
                        "project_count-to_save": len(new_projects),
                    },
                )
                return new_projects

        with tracer.start_as_current_span(
            "save_projects_to_db",
        ) as span, rx.session() as session:
            if not projects:
                span.add_event(
                    name="db-projects-no_projects_to_save",
                )
                return

            projects_to_save: list[Project] = filter_projects_to_save(
                db=session,
                projects=projects,
            )
            span.add_event(
                name="db-projects-projects_to_save",
                attributes={
                    "project_count": len(projects_to_save),
                },
            )
            if len(projects_to_save) == 0:
                span.add_event(
                    name="db-projects-no_new_projects_to_save",
                )
                if projects:
                    first_repo, *_ = projects
                    yield rx.toast.error(
                        f"Repo already saved: {first_repo.repo_path}",
                    )

                return

            session.bulk_save_objects(projects_to_save)
            session.commit()
            span.add_event(
                name="db-projects-added_to_db",
                attributes={
                    "project_count": len(projects_to_save),
                },
            )

    def save_project(
        self: State,
        project: Project,
    ) -> Generator[None, None, None]:
        with tracer.start_as_current_span("save_project") as span:
            self.projects_to_commit.append(project)
            span.add_event(
                name="projects_to_commit-added_project",
                attributes={
                    "project_repo_path": str(project.repo_path),
                },
            )
            yield from self._save_projects_to_db(
                projects=self.projects_to_commit,
            )

            span.add_event(
                name="projects_to_commit-saved_to_db",
            )
            self.projects_to_commit.clear()
            span.add_event(
                name="projects_to_commit-cleared",
            )

    def get_repo(
        self: State,
        repo_path: str,
    ) -> Repository | None:
        with tracer.start_as_current_span("get_repo") as span:
            repo_path_search: str = helper_github.extract_repo_path(
                repo_search_input=repo_path,
            )
            repo: Repository | None = helper_github.fetch_repo(
                repo_path=repo_path_search,
                client=CLIENT_GITHUB,
            )
            if repo is None:
                span.add_event(
                    name="repo-not_found",
                    attributes={
                        "repo_path": repo_path,
                    },
                )
                return None

            span.add_event(
                name="repo-found",
                attributes={
                    "repo_path": str(repo.full_name),
                    "repo_description": str(repo.description),
                },
            )
            return repo

    def generate_audio(
        self: State,
    ) -> None:
        self.has_generated_audio = True

    def _save_swot_to_db(
        self: State,
        swot: Swot,
    ) -> None:
        with tracer.start_as_current_span(
            "save_projects_to_db",
        ) as span, rx.session() as session:
            # Check if SWOT already exists for this repo
            existing_swot = (
                session.query(Swot).filter(Swot.repo_path == swot.repo_path).first()
            )
            if existing_swot:
                # Update existing SWOT
                if swot.strengths:
                    existing_swot.strengths = swot.strengths

                if swot.weaknesses:
                    existing_swot.weaknesses = swot.weaknesses

            else:
                # Add new SWOT
                session.add(swot)

            session.commit()
            session.refresh(swot)
            span.add_event(
                name="db-swots-added_to_db",
                attributes={
                    "swot_repo_path": str(swot.repo_path),
                    "operation": "update" if existing_swot else "insert",
                },
            )

    def save_swot(
        self: State,
        swot: Swot,
    ) -> None:
        with tracer.start_as_current_span("save_swot") as span:
            span.add_event(
                name="swot-added_to_db",
                attributes={
                    "swot_repo_path": str(swot.repo_path),
                },
            )
            self._save_swot_to_db(
                swot=swot,
            )
            span.add_event(
                name="swot-saved_to_db",
            )

    def get_strengths(
        self: State,
    ) -> None:
        repo_path: str | None = self.ag_grid_selection_repo_path
        if repo_path is None:
            return

        project_index: int | None = State._find_project_index_using_repo_path(
            projects=self.projects,
            repo_path=repo_path,
        )
        if project_index is None:
            return

        strengths: str = get_swot_from_exa_strenghts(
            organization_url=repo_path,
            client=CLIENT_EXA,
        )
        swot: Swot = Swot(
            repo_path=repo_path,
            strengths=strengths,
            weaknesses="",
        )
        self.save_swot(swot)

    def get_weaknesses(
        self: State,
    ) -> None:
        repo_path: str | None = self.ag_grid_selection_repo_path
        if repo_path is None:
            return

        project_index: int | None = State._find_project_index_using_repo_path(
            projects=self.projects,
            repo_path=repo_path,
        )
        if project_index is None:
            return

        project: Project = self.projects[project_index]
        weaknesses: str = get_swot_from_exa_weaknesses(
            organization_url=project.repo_url,
            client=CLIENT_EXA,
        )
        swot: Swot = Swot(
            repo_path=repo_path,
            strengths="",
            weaknesses=weaknesses,
        )
        self.save_swot(swot)

    @rx.background
    async def fetch_repo_and_submit(
        self: State,
    ):
        async with self:
            span = tracer.start_span("fetch_repo_and_submit")
            if self.repo_path_search == "":
                span.add_event(
                    name="fetch_repo_and_submit-no_repo_path_submitted",
                )
                yield rx.toast.error(
                    "no_repo_path_submitted",
                )
                return

            repo: Repository | None = self.get_repo(
                repo_path=self.repo_path_search,
            )
            self.clear_repo_path_search()
            if repo is None:
                yield rx.toast.error(
                    "repo-not_found",
                )
                return

            project: Project = Project.from_repo(
                repo=repo,
            )
            span.add_event(
                name="project-created_from_repo",
                attributes={
                    "project_repo_path": str(project.repo_path),
                    "project_stars": str(project.stars),
                    "project_language": str(project.language),
                    "project_website": str(project.website),
                    "project_description": str(project.description),
                    "project_created_at": str(project.created_at),
                },
            )
            self.add_project_to_display_data(project)
            self.clear_repo_path_search()
            yield

            # perplexity_description: str = await perplexity_get_repo(
            #     repo_url=project.repo_url,
            #     client=CLIENT_PERPLEXITY,
            # )
            # project.set_description(
            #     description=perplexity_description,
            # )
            # yield

            for item in self.save_project(project):
                yield item

            chroma_add_project(
                project=project,
                client=CLIENT_CHROMA,
            )

    def vector_search_filter(
        self: State,
    ) -> Generator[rx.Component, None, None]:
        with tracer.start_as_current_span("vector_search_filter") as span:
            span.add_event(
                name="vector_search_filter-called",
                attributes={
                    "repo_filter_text": str(self.last_vector_search_filter_text),
                },
            )
            distance_threshold: float = self.distance_threshold / 100
            project_repo_paths: list[str] = chroma_get_projects(
                repo_filter_vector_search_text=self.last_vector_search_filter_text,
                n_results=NUMBER_OF_RESULTS_TO_DISPLAY_FOR_VECTOR_SEARCH,
                client=CLIENT_CHROMA,
                distance_threshold=distance_threshold,
            )
            if project_repo_paths is None or not project_repo_paths:
                return

            self.display_data_indices = [
                index
                for index in (
                    self._find_project_index_using_repo_path(
                        projects=self.projects,
                        repo_path=repo_path,
                    )
                    for repo_path in project_repo_paths
                )
                if index is not None
            ]

    def on_load(
        self: State,
    ) -> None:
        with tracer.start_as_current_span("on_load") as span:
            with rx.session() as session:
                self.projects = (
                    session.exec(
                        statement=select(Project),
                    )
                    .scalars()
                    .all()
                )

            span.add_event(
                name="projects-loaded",
                attributes={
                    "project_count": len(self.projects),
                },
            )
            self.display_data_indices = list(range(len(self.projects)))

    def fetch_issues_for_repo(
        self: State,
        repo_path: str,
    ) -> Generator[None, None, None]:
        repo: Repository | None = helper_github.fetch_repo(
            repo_path=repo_path,
            client=CLIENT_GITHUB,
        )
        if repo is None:
            return

        yield from helper_github.fetch_github_issues_for_repo(
            repo=repo,
        )
