import dash
import json
from pathlib import Path
from dash import Dash, html, dcc, callback, html, Input, Output, register_page
from dash_iconify import DashIconify
import dash_mantine_components as dmc

# layout of Uni-RNA App
register_page(__name__, name="Contact Authors", path="/contact")

role_colors = {
    "First Author": "blue",
    "Corresponding Author": "pink",
    "Contributing Author": "gray",
}

authors = json.loads((Path(__file__).parent / "authors.json").read_text())

def generate_personal_card(name: str, mail: str, role: str, avatar: str):
    return (
        dmc.Card(
            children=[
                dmc.Group(
                    [
                        dmc.Avatar(
                            src=avatar   
                        ),
                        dmc.Text(name, fw=500),
                        dmc.Badge(role, color=role_colors.get(role, "gray"))
                    ],
                    mt="md",
                    mb="xs",
                ),
                dmc.Text(
                    [
                        DashIconify(
                            icon="line-md:email-twotone", 
                            style={
                                "marginRight": "0.5rem",
                            }
                        ),
                        mail
                    ], 
                    fw=500
                ),
            ]
        )
    )

layout = [
    dmc.Container(
        children=[
            generate_personal_card(
                name=author,
                mail=v["email"],
                role=v["role"],
                avatar=v["avatar"],
            )
            for author, v in authors.items()
        ],
    )
]