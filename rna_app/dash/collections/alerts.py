import dash_mantine_components as dmc
from dash_iconify import DashIconify


no_input_alert = dmc.Alert(
    title="Please provide input",
    color="yellow",
    icon=dmc.ActionIcon(
        DashIconify(icon="line-md:loading-alt-loop"),
        size="md",
        variant="filled",
        color="yellow",
    ),
)

standby_alert = dmc.Alert(
    title=[
        "Read to go!",
    ],
    color="grey",
    icon=dmc.ActionIcon(
        DashIconify(icon="line-md:speed-loop"),
        size="md",
        variant="filled",
        color="grey",
    ),
)

success_alert = dmc.Alert(
    title="Success!",
    color="green",
    icon=dmc.ActionIcon(
        DashIconify(icon="line-md:check-all"),
        size="md",
        variant="filled",
        color="green",
    ),
)

fail_alert = dmc.Alert(
    children=f"Please check your input. It should be in FASTA format. If you are sure that your input is correct, please contact the developer.",
    title="Error",
    color="red",
    icon=dmc.ActionIcon(
        DashIconify(icon="line-md:emoji-frown"),
        size="md",
        variant="filled",
        color="red",
    ),
)

duplicated_name_alert = dmc.Alert(
    children="Duplicated name found. Please provide unique names for each sequence.",
    title="Error",
    color="red",
    icon=dmc.ActionIcon(
        DashIconify(icon="line-md:emoji-frown"),
        size="md",
        variant="filled",
        color="red",
    ),
)