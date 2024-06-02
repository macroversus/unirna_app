import dash_mantine_components as dmc
from dash_iconify import DashIconify


wrong_input = dmc.Notification(
    title="Hey there!",
    id="simple-notify",
    action="show",
    message="Notifications in Dash, Awesome!",
    icon=DashIconify(icon="ic:round-celebration"),
)
