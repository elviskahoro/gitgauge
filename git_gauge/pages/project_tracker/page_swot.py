from __future__ import annotations

import reflex as rx
from reflex_ag_grid import ag_grid

from git_gauge.models.project import Project
from git_gauge.pages.project_tracker.constants import (
    AG_GRID_ID,
    AG_GRID_THEME,
    REPO_SEARCH_INPUT_ID,
)
from git_gauge.pages.project_tracker.state import State

AG_GRID_COLUMN_DEFINITIONS = Project.get_ag_grid_column_definitions()


def index() -> rx.Component:
    return rx.vstack(
        rx.hstack(
            rx.logo(),
            rx.color_mode.button(),
            margin_y="1em",
            width="100%",
            justify_content="space-between",
        ),
        rx.vstack(
            rx.fragment(State.repo_card_strength),
            rx.fragment(State.repo_card_weakness),
            rx.fragment(State.repo_card_stats),
        ),
        rx.flex(
            rx.input(
                id=REPO_SEARCH_INPUT_ID,
                value=State.repo_path_search,
                placeholder="GitHub repo path e.g. reflex-dev/reflex",
                on_change=State.setter_repo_path_search,
                width="40%",
            ),
            rx.hstack(
                rx.button(
                    "Fetch repo data",
                    on_click=State.fetch_repo_and_submit,
                    margin_bottom="1em",
                ),
                rx.button(
                    "Get strengths",
                    on_click=State.get_strengths,
                    margin_bottom="1em",
                ),
                rx.button(
                    "Get weaknesses",
                    on_click=State.get_weaknesses,
                    margin_bottom="1em",
                ),
                rx.button(
                    "Generate audio",
                    on_click=State.generate_audio,
                    margin_bottom="1em",
                ),
            ),
            justify_content="space-between",
            width="100%",
        ),

        rx.flex(
            rx.cond(
                State.has_generated_audio,
                rx.audio(
                    id="swot-audio",
                    url=rx.get_upload_url(State.audio_file_path),
                    height="32px",
                ),
                rx.skeleton(
                    rx.audio(
                        id="swot-audio-skeleton",
                        height="32px",
                    ),
                ),
            ),
            max_height="30px",
        ),
        ag_grid(
            id=AG_GRID_ID,
            column_defs=AG_GRID_COLUMN_DEFINITIONS,
            row_data=State.display_data,
            pagination=True,
            pagination_page_size=100,
            pagination_page_size_selector=[
                50,
                100,
            ],
            on_selection_changed=State.setter_repo_path_ag_grid_selection,
            theme=AG_GRID_THEME,
            width="100%",
            height="60vh",
        ),
        width="80%",
        margin="0 auto",
        spacing="4",
    )
