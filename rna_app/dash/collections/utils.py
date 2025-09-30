from pathlib import Path
from Bio import SeqIO
import os
import base64
import io
from datetime import datetime
from dash import (
    Dash,
    html,
    dcc,
    callback,
    Output,
    Input,
    clientside_callback,
    register_page,
    dash_table,
)
import dash_mantine_components as dmc
import dash_ag_grid as dag
from rna_app.dash.collections.alerts import *

os.environ["REACT_VERSION"] = "18.2.0"

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "rna_app_outputs"
EXAMPLE_DIR = PROJECT_ROOT / "example"

example_fastas = {
    "acceptor": str(EXAMPLE_DIR / "acceptor" / "input.fasta"),
    "donor": str(EXAMPLE_DIR / "donor" / "input.fasta"),
    "utr": str(EXAMPLE_DIR / "utr" / "input.fasta"),
    "m6a": str(EXAMPLE_DIR / "m6a" / "input.fasta"),
    "rna_ss": str(EXAMPLE_DIR / "unirna_ss" / "input.fasta"),
    "pirna": str(EXAMPLE_DIR / "pirna" / "input.fasta"),
    "lncrna_sublocalization": str(
        EXAMPLE_DIR / "lncrna_sublocalization" / "input.fasta"
    ),
    "extract_embedding": str(EXAMPLE_DIR / "extract_embedding" / "input.fasta"),
    "apa": str(EXAMPLE_DIR / "apa" / "input.fasta"),  # Use UTR example for APA
}

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
            leftSection=DashIconify(icon="line-md:upload-outline-loop"),
        ),
        dmc.Text("Max file size: 1 MB"),
    ],
    max_size=11485760,
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

status = dmc.Container(
    id="status",
    ml="md",
)

ag_table = dag.AgGrid(
    id="result-table",
    columnSize="sizeToFit",
    columnDefs=[],
    rowData=[],
    csvExportParams=None,
)
output_table = html.Div(
    id="output-table",
    children=[
        # dmc.Button("Export to CSV", id="export-csv", n_clicks=0),
        ag_table,
    ],
    style={
        "marginBottom": "50px",
    },
    hidden=True,
)


# @callback(
#     Output("result-table", "exportDataAsCsv"),
#     Input("export-csv", "n_clicks"),
# )
# def export_csv(n_clicks):
#     if n_clicks:
#         return True
#     return False

@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Output("upload-fasta", "contents"),
    Output("status", "children", allow_duplicate=True),
    Input("upload-fasta", "contents"),
    Input("upload-fasta", "last_modified"),
    prevent_initial_call=True,
)
def load_from_fasta_file(contents, last_modified):
    if not contents:
        return ""
    fasta_text = ""
    for content in contents:
        _, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        for record in SeqIO.parse(io.StringIO(decoded.decode()), "fasta"):
            fasta_text = f"{fasta_text}>{record.description}\n{record.seq}\n"
    return fasta_text, None, standby_alert


@callback(
    Output("status", "children", allow_duplicate=True),
    Input("fasta-text", "value"),
    prevent_initial_call=True,
)
def check_fasta_text(fasta_text: str):
    if fasta_text:
        seq_names = []
        for record in SeqIO.parse(io.StringIO(fasta_text), "fasta"):
            seq_name = record.description
            if seq_name in seq_names:
                return (
                    "",
                    None,
                    duplicated_name_alert
                )
            seq_names.append(seq_name)
        return standby_alert
    else:
        return no_input_alert

def get_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")