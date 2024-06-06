import dash
from dash import Dash, html, dcc, callback, html, Input, Output, register_page
from dash_iconify import DashIconify
import dash_mantine_components as dmc

# layout of Uni-RNA App
register_page(__name__, name="Uni-RNA Apps", path="/")

main_layout = dmc.BackgroundImage(
    children=dmc.Center(
        children=[
            dmc.Text(
                children=[
                    dmc.Text(
                        [
                            DashIconify(
                                icon="line-md:chat", 
                                style={
                                    "marginRight": "0.5rem",
                                }
                            ), 
                            "This is an inner page for reviewers to evaluate Uni-RNA models."
                        ], 
                        style={"fontSize": "1.5rem",}),
                    dmc.Text("We provide easy-to-use webserver for all the major downstream tasks in our paper.", style={"fontSize": "1.5rem"}),
                ],
                style={
                    "color": "white",
                    "textAlign": "center",
                    "margin": "2rem",
                    "backgroundColor": "rgba(0, 0, 0, 0.5)",
                }
            ),
            dmc.Button(
                children=dmc.Anchor("Explore Our Uni-RNA Apps !", href="/apps", size="xl", style={"color": "white", "cursor": "pointer"}),
                size="md", 
                variant="gradient",
                gradient={"from": "indigo", "to": "cyan"},
                leftSection=DashIconify(icon="svg-spinners:blocks-scale")
            ),
        ],
        display="flex",
        style={
            "height": "100%", 
            "width": "100%",
            "flexDirection": "column",
        },
    ),
    src="/assets/main.png",
    h=800,
    w="100%",
    radius="md",
)

layout = [
    dmc.Container(
        children=[
            main_layout,
        ],
        size="xl",
    )
]