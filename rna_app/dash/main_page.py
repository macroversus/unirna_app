import dash
from dash import Dash, html, dcc, callback, html, Input, Output
from dash_iconify import DashIconify
import dash_mantine_components as dmc

app = Dash(__name__, use_pages=True, pages_folder="pages")
server = app.server

menu = dmc.Menu(
    [
        dmc.MenuTarget(dmc.Button("Click to See More Uni-RNA Apps!")),
        dmc.MenuDropdown(
            [
                dmc.MenuItem(
                    page['name'],
                    href=page["relative_path"],
                    leftSection=DashIconify(icon="radix-icons:external-link"),
                )
                for page in dash.page_registry.values()
            ],
        ),
    ],
)

app.layout = dmc.MantineProvider(
    children=[
        html.H1("Uni-RNA App", style={"textAlign": "center"}),
        dmc.Container(menu, style={"justifyContent": "center", "margin": "auto"}),
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=50004, host="0.0.0.0")
