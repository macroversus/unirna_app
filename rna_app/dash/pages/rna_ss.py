import pandas as pd
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from datetime import datetime
from uuid import uuid4
from tempfile import TemporaryDirectory
import subprocess
from dash import (
    Dash,
    html,
    dcc,
    callback,
    Output,
    Input,
    clientside_callback,
    register_page,
    State,
    ctx,
)
from dash.exceptions import PreventUpdate
import dash_bio
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import (
    no_input_alert,
    standby_alert,
    success_alert,
    fail_alert,
)

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
            span="2",
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
        {
            "value": "unirna",
            "label": "Type-1",
        },
        {
            "value": "archiveii",
            "label": "Type-2",
        },
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

ss_store = dcc.Store(id="ss-store", storage_type="session")
ss_contrainer = dash_bio.FornaContainer(id="ss-contrainer", height=400, width=800)
ss_display = dcc.Dropdown(id="ss-display", options=[], multi=True, value=[])

download_ss = dmc.Button(
    "Export Results to CSV",
    id="export-csv_rna_ss",
    n_clicks=0,
)

outputs_ss = dmc.Container(
    id="outputs-ss",
    children=[
        html.Hr(),
        ss_store,
        html.P('Select the sequences to display below.'),
        ss_display,
        html.Hr(),
        ss_contrainer,
        html.Hr(),
        download_ss
    ],
    w="100%",
    display="none",
)


@callback(
    Output("result-table", "exportDataAsCsv", allow_duplicate=True),
    Output("result-table", "csvExportParams", allow_duplicate=True),
    Input("export-csv_rna_ss", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(n_clicks):
    print(ctx.triggered_id)
    if ctx.triggered_id == "export-csv_rna_ss":
        return True, {
            "fileName": f"rna_ss_results_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    else:
        return False, {}

@callback(
    Output("ss-contrainer", "sequences"),
    Input("ss-display", "value"),
    State("ss-store", "data"),
    prevent_initial_call=True,
)
def show_selected_sequences(value, data):
    if value is None:
        raise PreventUpdate
    return [data[name] for name in value]


@callback(
    Output("start-button-rna_ss", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("outputs-ss", "display", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("ss-display", "options", allow_duplicate=True),
    Output("ss-display", "value", allow_duplicate=True),
    Output("ss-store", "data", allow_duplicate=True),
    Input("start-button-rna_ss", "loading"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
)
def start_infer_rna_ss(loading: bool, fasta_text: str):
    if not fasta_text:
        return False, [], [], True, "none", no_input_alert, [], [], []
    if loading:
        try:
            with TemporaryDirectory() as temp_dir:
                in_fasta = f"{temp_dir}/input.fasta"
                with open(in_fasta, "w") as f:
                    f.write(fasta_text)
                process_ret = subprocess.run(
                    [
                        "rna_app_infer",
                        "--in_data",
                        in_fasta,
                        "--mission",
                        "rna_ss",
                        "--output_dir",
                        temp_dir,
                    ]
                )
                assert process_ret.returncode == 0, "Inference failed"
                ret = pd.read_csv(f"{temp_dir}/result.csv")
            return (
                False,
                ret.to_dict("records"),
                [{"field": i} for i in ret.columns],
                True,
                "inline-block",
                success_alert,
                ret["name"].tolist(),
                [ret["name"].iloc[0]],
                {
                    name: {
                        "sequence": seq,
                        "structure": ss,
                        "options": {
                            "name": name,
                            "applyForce": False
                        }
                    }
                    for name, seq, ss in ret[
                        ["name", "seq", "secondary_structure"]
                    ].values
                },
            )
        except Exception as e:
            return False, [], [], True, "none", [fail_alert, f"Error: {e}"], [], [], []


layout = [
    html.Div(
        children=[
            html.Div("RNA Secondary Structure Prediction", style={"flex": "1", "textAlign": "center"}),
        ],
        style={
            "textAlign": "center", 
            "fontSize": 30, 
            "display": "flex",
            "alignItems": "center",
        },
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                upload_fasta,
                model_type_selection,
                rna_ss_fasta_input,
                start_button_rna_ss,
                outputs_ss,
                output_table,
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
