import dash
from dash import Dash, html, dcc, callback, html, Input, Output, register_page
from dash_iconify import DashIconify
import dash_mantine_components as dmc


register_page(__name__, name="Uni-RNA Copyright", path="/copyright")

layout = [
    dmc.Container(
        children=[
            dmc.Title("Copyright Notice", order=1, mb="lg"),
            dmc.Text(r"Institute for Advanced Algorithms Research, Shanghai. (hereinafter referred to as “IAAR”) fully possesses the copyright of this code.",),
            dmc.Text(r"Without IAAR’s written authorization, any natural person or enterprise shall not copy, forward or perform any unauthorized act. If authorized, the source must be clearly indicated.", ),
            dmc.Text(r"Otherwise, IAAR has the right to pursue the liability for the violation of the above mentioned terms, and reserves the right to pursue other legal responsibilities.", ),
        ],
        size="xl",
    )
]