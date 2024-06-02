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
from dash.exceptions import PreventUpdate
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import no_input_alert, standby_alert, success_alert, fail_alert
from rna_app.core.m6a import infer_m6a

register_page(__name__)

start_button_m6a = dmc.Grid(
    children=[
        dmc.GridCol(
        dmc.Button(
            id="start-button-m6a",
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

fetch_example_m6a = dmc.Button(
    id="fetch-example-m6a",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

urt_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_m6a, span="2"),
    ],
    align="flex-end",
)

@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-m6a", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_m6a(n_clicks):
    fasta_text = ""
    for record in SeqIO.parse(example_fastas["m6a"], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-m6a", "loading", allow_duplicate=True),
    Input("start-button-m6a", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("start-button-m6a", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("result-table", "csvExportParams", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-m6a", "loading"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
)
def start_infer_m6a(loading: bool, fasta_text: str):
    print(datetime.now(), "m6a")
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
                        "--mission", "m6a",
                        "--output_dir", temp_dir,
                    ]
                )
                assert process_ret.returncode == 0, "Inference failed"
                ret = pd.read_csv(f"{temp_dir}/result.csv")
            return False, ret.to_dict("records"), [{"field": i} for i in ret.columns], {"fileName": f"m6a_results_{now}.csv"}, False, success_alert
        except Exception as e:
            return False, [], [], None, True, [fail_alert, f"Error: {e}"]

layout = [
    html.Div(children="5' m6a Mean Ribosomal Load Prediction", style={"textAlign": "center", "fontSize": 30}),
    html.Hr(),
    dmc.MantineProvider(
        children = dmc.Container(
            children = [
                upload_fasta,
                urt_fasta_input,
                start_button_m6a,
                output_table,
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
