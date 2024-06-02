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
from rna_app.core.acceptor import infer_acceptor
from rna_app.core.donor import infer_donor

register_page(__name__)

acceptor_or_donor = dmc.RadioGroup(
    id="splice-type",
    children=[
        "Select the type of splice site you want to infer:",
        dmc.Group(
            [
                dmc.Radio(label="Acceptor", value="acceptor"),
                dmc.Radio(label="Donor", value="donor"),
            ],
        ),
    ],
    value="acceptor",
    style={"display": "flex", "margin": "20px"},
    readOnly=False,
)

start_button_splicesite = dmc.Grid(
    children=[
        dmc.GridCol(
        dmc.Button(
            id="start-button-splicesite",
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

fetch_example_splicesite = dmc.Button(
    id="fetch-example-splicesite",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

urt_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_splicesite, span="2"),
    ],
    align="flex-end",
)

@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-splicesite", "n_clicks"),
    State("splice-type", "value"),
    prevent_initial_call=True,
)
def fetch_example_splicesite(n_clicks, splice_type):
    fasta_text = ""
    for record in SeqIO.parse(example_fastas[splice_type], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-splicesite", "loading", allow_duplicate=True),
    Input("start-button-splicesite", "n_clicks"),
    prevent_initial_call=True,
)

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("splice-type", "readOnly", allow_duplicate=True),
    Input("start-button-splicesite", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("start-button-splicesite", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("result-table", "csvExportParams", allow_duplicate=True),
    Output("splice-type", "readOnly", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-splicesite", "loading"),
    Input("splice-type", "value"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
)
def start_infer_splicesite(loading: bool, splice_type: str, fasta_text: str):
    print(datetime.now(), "splice_type:", splice_type)
    if not fasta_text:
        return False, [], [], None, False, True, no_input_alert
    if loading:
        try:
            now = datetime.now().strftime('%Y%m%d_%H%M%S')
            with TemporaryDirectory() as temp_dir:
                in_fasta = f"{temp_dir}/input.fasta"
                outfile = f"{temp_dir}/result.csv"
                with open(in_fasta, "w") as f:
                    f.write(fasta_text)
                process_ret = subprocess.run(
                    [
                        "rna_app_infer",
                        "--in_data", in_fasta,
                        "--mission", splice_type,
                        "--output_dir", temp_dir,
                    ]
                )
                assert process_ret.returncode == 0, "Inference failed"
                ret = pd.read_csv(outfile)
            return False, ret.to_dict("records"), [{"field": i} for i in ret.columns], {"fileName": f"{splice_type}_results_{now}.csv"}, False, False, success_alert
        except Exception as e:
            return False, [], [], None, False, True, [f"Error: {e}", fail_alert]
    else:
        return False, [], [], None, False, True, standby_alert

layout = [
    html.Div(children="Splice Site", style={"textAlign": "center", "fontSize": 30}),
    html.Hr(),
    dmc.MantineProvider(
        children = dmc.Container(
            children = [
                upload_fasta,
                urt_fasta_input,
                dmc.MantineProvider(acceptor_or_donor),
                start_button_splicesite,
                output_table,
            ],
            style={
                "width": "80%",
                "marginBottom": "50px",
            }
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
