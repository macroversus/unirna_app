import pandas as pd
from Bio import SeqIO
from datetime import datetime
import io
import dash_mantine_components as dmc
from datetime import datetime
from uuid import uuid4
from tempfile import TemporaryDirectory
import subprocess
from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, register_page, State
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import no_input_alert, standby_alert, success_alert, fail_alert

register_page(__name__, name="RNA Secondary Structure Prediction", path="/rna_ss")

start_button_rna_ss = dmc.Grid(
    children=[
        dmc.GridCol(
        dmc.Button(
            id="start-button-rna_ss",
            children="Start Inference",
            radius="md",
            style={
                "marginTop": "10px",
                "marginBottom": "10px",
            },
            loaderProps={"type": "dots"}, 
            loading=False,
            leftSection=DashIconify(icon="line-md:play-twotone"),
        ),
        span="2"
        ),
        dmc.GridCol(status, span="10"),
    ],
    align="flex-start",
)

fetch_example_rna_ss = dmc.Button(
    id="fetch-example-rna_ss",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

rna_ss_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_rna_ss, span="2"),
    ],
    align="flex-end",
)

@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-rna_ss", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_rna_ss(n_clicks):
    fasta_text = ""
    for record in SeqIO.parse(example_fastas["rna_ss"], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text

model_type = dmc.Select(
    label="Model Type",
    placeholder="Select one",
    id="model-type",
    value="unirna",
    data=[
        {"value": "unirna", "label": "Type-1", },
        {"value": "archiveii", "label": "Type-2", },
    ],
    allowDeselect=False,
)
    
model_type_selection = dmc.Grid(
    children=[
        dmc.GridCol(model_type, span="2"),
        dmc.GridCol(dmc.Alert(id="model-type-description", color="blue"), span="10"),
    ],
    align="center",
)

@callback(Output("model-type-description", "children"), Input("model-type", "value"))
def select_value(value):
    if value == "unirna":
        return "Trained on the Uni-RNA 50% threshold secondary structure dataset"
    else:
        return "Trained on the MXfold2 (RNAStralign) dataset"

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-rna_ss", "loading", allow_duplicate=True),
    Input("start-button-rna_ss", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("start-button-rna_ss", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("result-table", "csvExportParams", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-rna_ss", "loading"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
)
def start_infer_rna_ss(loading: bool, fasta_text: str):
    if not fasta_text:
        return False, [], [], None, True, no_input_alert
    if loading:
        try:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            with TemporaryDirectory() as temp_dir:
                in_fasta = f"{temp_dir}/input.fasta"
                with open(in_fasta, "w") as f:
                    f.write(fasta_text)
                process_ret = subprocess.run(
                    [
                        "rna_app_infer",
                        "--in_data", in_fasta,
                        "--mission", "rna_ss",
                        "--output_dir", temp_dir,
                    ]
                )
                assert process_ret.returncode == 0, "Inference failed"
                ret = pd.read_csv(f"{temp_dir}/result.csv")
            return False, ret.to_dict("records"), [{"field": i} for i in ret.columns], {"fileName": f"rna_ss_results_{now}.csv"}, False, success_alert
        except Exception as e:
            return False, [], [], None, True, [fail_alert, f"Error: {e}"]

layout = [
    html.Div(children="RNA Secondary Structure Prediction", style={"textAlign": "center", "fontSize": 30}),
    html.Hr(),
    dmc.MantineProvider(
        children = dmc.Container(
            children = [
                upload_fasta,
                model_type_selection,
                rna_ss_fasta_input,
                start_button_rna_ss,
                output_table,
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
