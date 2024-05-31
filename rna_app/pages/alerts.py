import dash_mantine_components as dmc
from dash_iconify import DashIconify


no_input_alert = dmc.Alert(
    children="No Input Found!",
    title="Please provide input",
    color="yellow"
)

standby_alert = dmc.Alert(
    children=[
        "Read to go !",
    ],
    title=[
    dmc.ActionIcon(
        DashIconify(icon="line-md:speed-loop", width=20),
        size="lg",
        variant="filled",
        id="action-icon",
        n_clicks=0,
        mb=10,
    ),
    ],
    color="grey"
)

success_alert = dmc.Alert(
    children="Success!",
    title="Success",
    color="green"
)