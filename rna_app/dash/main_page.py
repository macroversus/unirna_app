import os
os.environ["REACT_VERSION"] = "18.2.0"
import dash
from dash import Dash, html, dcc, callback, html, Input, Output
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import argparse

app = Dash(__name__, use_pages=True, pages_folder="pages")
server = app.server

navbar = dmc.Container(
    children=[
        dmc.Anchor(
            href="/",
            children=[
                    dmc.Button(
                        "Home",
                        fw=700,
                        size="md",
                    ),
                ],
            ),
        dmc.Group(
            children=[
                dmc.Anchor(dmc.Button("Explore Apps"), href="/apps", size="md"),
                dmc.Anchor(dmc.Button("Contact"), href="/contact", size="md"),
                dmc.Anchor(dmc.Button("Copyright"), href="/copyright", size="md"),
            ],
            style={"marginLeft": "auto"},
        ),        
    ],
    style={
        "marginTop": "2%",  
    },
    display="flex",
    size="xl",
)

app.layout = dmc.MantineProvider(
    children=[
        navbar,
        dmc.Divider(label="Molecule Simulates the Future", labelPosition="center", variant="dashed", size="xs", color="blue"),
        html.H1("Uni-RNA Apps", style={"textAlign": "center"}),
        dash.page_container,
    ],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Uni-RNA Dash App")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    parser.add_argument("--port", type=int, default=50004, help="Port to run the server on (default: 50004)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()
    
    app.run_server(debug=args.debug, port=args.port, host=args.host)
