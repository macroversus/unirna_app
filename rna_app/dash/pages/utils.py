from pathlib import Path
from Bio import SeqIO
import os
import base64
import io
from datetime import datetime
from time import sleep
from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, register_page, dash_table
import plotly.express as px
import dash_mantine_components as dmc
from dash_iconify import DashIconify

os.environ["REACT_VERSION"] = "18.2.0"

OUTPUT_ROOT = Path(__file__).parent.parent.parent / "rna_app_outputs"

upload_fasta = dcc.Upload(
    id="upload-fasta",
    children=[
        dmc.Text("Drag and Drop or", style={"display": "inline-block"}),
        dmc.Button(
            "Select FASTA Files",
            radius="md",
            style={
                "display": "inline-block",
                "borderWidth": "0px",
                "fullWidth": True,
            },
        ),
        dmc.Text("Max file size: 1MB"),
    ],
    max_size=1e6,
    multiple=True,
    style={
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
    },
)

fasta_textarea = dmc.Textarea(
    id="fasta-text",
    label="Sequence in FASTA format",
    placeholder="You can also paste your sequences in FASTA format here.",
    autosize=True,
    required=True,
    maxRows=16,
)

start_button = dmc.Button(
    id="start-button",
    children="Start Inference",
    radius="md",
    style={
        "margin": "10px",
    },
    loaderProps={"type": "dots"}, 
    loading=False,
)

status = dmc.Container(
    id="status",
)

output_table = dash_table.DataTable(
        id='result-table',
        columns=[],
        data=[],
        editable=False,
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_size= 10,
        style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
        export_format=None,
)

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button", "loading", allow_duplicate=True),
    Input("start-button", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("fasta-text", "value"),
    Output("upload-fasta", "contents"),
    Input("upload-fasta", "contents"),
    Input("upload-fasta", "last_modified"),
    prevent_initial_call=True,
)
def load_from_fasta_file(contents, last_modified):
    print(contents)
    if not contents:
        return ""
    fasta_text = ""
    for content in contents:
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        for record in SeqIO.parse(io.StringIO(decoded.decode()), "fasta"):
            fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text, None
