import dash
from dash import Dash, html, dcc, callback, html, Input, Output, register_page
from dash_iconify import DashIconify
import dash_mantine_components as dmc

# layout of Uni-RNA App
register_page(__name__, name="Main", path="/")
layout = []