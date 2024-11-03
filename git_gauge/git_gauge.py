# trunk-ignore-all(trunk/ignore-does-nothing)
import reflex as rx

from .app_style import Style as AppStyle
from .pages.project_tracker.page_swot import index as index_swot

# from .pages.project_tracker.page_vector_search import index as index_vector_search
from .pages.project_tracker.state import State

APP_STYLE: AppStyle = AppStyle()

app = rx.App(
    style=APP_STYLE.dark,
    theme=rx.theme(
        has_background=True,
        radius="large",
        accent_color="teal",
    ),
)
app.add_page(
    component=index_swot,
    route="/",
    on_load=State.on_load, # trunk-ignore(pyright/reportArgumentType)
)
