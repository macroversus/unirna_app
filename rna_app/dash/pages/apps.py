import dash
import json
from pathlib import Path
from dash import Dash, html, dcc, callback, html, Input, Output, register_page
from dash_iconify import DashIconify
import dash_mantine_components as dmc

# layout of Uni-RNA App
register_page(__name__, name="Explore Uni-RNA Apps", path="/apps")

def generate_card(app_name: str, app_description: str, app_href: str, app_figure: str):
    return (
        dmc.Card(
                children=[
                    dmc.CardSection(
                        dmc.Image(
                            src=app_figure,
                            h=160,
                            alt=app_name,
                            style={
                                "borderBlockEnd": "1px solid #e1e1e1",
                            }
                        )
                    ),
                    dmc.Group(
                        [
                            dmc.Text(app_name, fw=500),
                        ],
                        justify="space-between",
                        mt="md",
                        mb="xs",
                    ),
                    dmc.ScrollArea(
                        app_description,
                        c="dimmed",
                        h=120,
                        offsetScrollbars=True,
                        ta="justify",
                    ),
                    dmc.NavLink(
                        label="Try App !",
                        leftSection=DashIconify(icon="line-md:chevron-right-circle", width=20, height=20),
                        color="blue",
                        active=True,
                        variant="filled",
                        href=app_href,
                        w="45%",
                        mt="md",
                        mr="auto",
                        fw=750,
                        ta="justify",
                        style={
                            "borderRadius": "0.5rem",
                        }
                    ),
                ],
                withBorder=True,
                shadow="sm",
                radius="md",
                w=350,
            )
    )


app_config = Path(__file__).parent / "config.json"
app_data = json.loads(app_config.read_text())

layout = [
    dmc.Container(
        children=dmc.Grid(
            children=[
                dmc.GridCol(
                    children=generate_card(
                        app_name=v["page_name"],
                        app_description=v["description"],
                        app_href=v["href"],
                        app_figure=v["fig"],
                    ),
                    span=4,
                )
                for v in app_data.values()
            ],
        ),
        size="xl",
    )
]