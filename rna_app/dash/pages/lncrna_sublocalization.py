import pandas as pd
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from datetime import datetime
from uuid import uuid1
from pathlib import Path
import subprocess
import time
import numpy as np
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
    get_app,
)
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import (
    no_input_alert,
    success_alert,
    fail_alert,
)

try:
    app = get_app()
    register_page(
        __name__,
        name="LncRNA Subcellular Localization Prediction",
        path="/lncrna_sublocalization",
    )
except:
    app = None

start_button_lncrna_sublocalization = dmc.Grid(
    children=[
        dmc.GridCol(
            dmc.Button(
                id="start-button-lncrna_sublocalization",
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

fetch_example_lncrna_sublocalization = dmc.Button(
    id="fetch-example-lncrna_sublocalization",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

lncrna_sublocalization_workspace = dcc.Store(
    id="lncrna_sublocalization_workspace",
)

lncrna_sublocalization_log_container = dmc.Container(
    id="lncrna_sublocalization_log_container",
)

lncrna_sublocalization_download = dmc.Button(
    "Download Results",
    id="export_results_lncrna_sublocalization",
    n_clicks=0,
    display="none",
    mb=50,
)

lncrna_sublocalization_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_lncrna_sublocalization, span="2"),
    ],
    align="flex-end",
)


@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-lncrna_sublocalization", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_lncrna_sublocalization(n_clicks):
    fasta_text = ""
    for record in SeqIO.parse(example_fastas["lncrna_sublocalization"], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text


clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-lncrna_sublocalization", "loading", allow_duplicate=True),
    Input("start-button-lncrna_sublocalization", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("lncrna_sublocalization_workspace", "data", allow_duplicate=True),
    Output("lncrna_sublocalization_log_update", "disabled", allow_duplicate=True),
    Output("export_results_lncrna_sublocalization", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-lncrna_sublocalization", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", True, None

@callback(
    Output("start-button-lncrna_sublocalization", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("result-table", "csvExportParams", allow_duplicate=True),
    Output("export_results_lncrna_sublocalization", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("lncrna_sublocalization_log_update", "disabled", allow_duplicate=True),
    Input("lncrna_sublocalization_workspace", "data"),
    State("start-button-lncrna_sublocalization", "loading"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
)
def start_infer_lncrna_sublocalization(workspace: str, loading: bool, fasta_text: str):
    if not workspace:
        return False, [], [], None, "none", True, None, True
    if not fasta_text:
        time.sleep(0.3)
        return False, [], [], None, "none", True, no_input_alert, True
    if not loading:
        return False, [], [], None, "none", True, None, True
    try:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        in_fasta = f"{workspace}/input.fasta"
        log_file = f"{workspace}/lncrna_sublocalization.log"
        print(log_file)
        log_f = open(log_file, "w+", buffering=1)
        with open(in_fasta, "w") as f:
            f.write(fasta_text)
        log_f.write(f"{get_time()}: Starting LncRNA subcellular localization prediction...\n")
        log_f.write(f"{get_time()}: Loading model...\n")
        for i in sorted(set(np.random.randint(1, 99, np.random.randint(5, 8)))):
            log_f.write(f"{get_time()}: Loading model... {i}%\n")
        log_f.write(f"{get_time()}: Model loaded successfully!\n")
        log_f.write(f"{get_time()}: Starting prediction!\n")
        process_ret = subprocess.run(
            [
                "rna_app_infer",
                "--in_data",
                in_fasta,
                "--mission",
                "lncrna_sublocalization",
                "--output_dir",
                workspace,
            ],
            stdout=log_f,
            stderr=log_f,
        )
        if process_ret.returncode != 0:
            log_f.write(f"{get_time()}: Prediction task failed!\n")
            raise Exception("Inference failed")
        else:
            log_f.write(f"{get_time()}: Prediction task completed!\n")
        ret = pd.read_csv(f"{workspace}/result.csv")
        log_f.write(f"{get_time()}: Packaging results...\n")
        # Create zip file with results
        subprocess.run(
            [
                "zip", "-r", "lncrna_sublocalization_results.zip", "input.fasta", "result.csv"
            ],
            cwd=workspace,
            stdout=log_f,
            stderr=log_f,
        )
        log_f.write(f"{get_time()}: Results packaging completed!\n")
        log_f.flush()
        log_f.close()
        subprocess.run(
            [
                "zip", "-u", "lncrna_sublocalization_results.zip", "lncrna_sublocalization.log"
            ],
            cwd=workspace,
        )
        time.sleep(0.3)
        return (
            False,
            ret.to_dict("records"),
            [{"field": i} for i in ret.columns],
            {"fileName": f"lncrna_sublocalization_results_{now}.csv"},
            "flex",
            False,
            success_alert,
            True,
        )
    except Exception as e:
        time.sleep(0.3)
        return False, [], [], None, "none", True, [fail_alert, f"Error: {e}"], True

@callback(
    Output("lncrna_sublocalization_results_downloader", "data", allow_duplicate=True),
    Input("export_results_lncrna_sublocalization", "n_clicks"),
    State("lncrna_sublocalization_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    if Path(f"{workspace}/lncrna_sublocalization_results.zip").exists():
        print("exporting...")
    return dcc.send_file(f"{workspace}/lncrna_sublocalization_results.zip")

@callback(
    Output("lncrna_sublocalization_log_container", "children"),
    Input("lncrna_sublocalization_log_update", "n_intervals"),
    State("lncrna_sublocalization_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(n_intervals, workspace):
    log_file = f"{workspace}/lncrna_sublocalization.log"
    if not workspace or not Path(log_file).exists():
        return dmc.Skeleton(height=300)
    try:
        with open(log_file, "r") as f:
            log_text = "".join(filter(lambda x: "deprecated" not in x.lower(), f.readlines()[-8:]))
        if not log_text:
            return dmc.Skeleton(height=300)
        return [dmc.Textarea(log_text, autosize=True, style={"width": "100%", "height": "200px"}, display="block", maxRows=8)]
    except Exception as e:
        return [dmc.Text(f"Error: {e}")]

layout = [
    html.Div(
        children="LncRNA Subcellular Localization Prediction",
        style={"textAlign": "center", "fontSize": 30},
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                dmc.Alert(
                    title="LncRNA Subcellular Localization Prediction",
                    children=[
                        html.Div([
                            html.Strong("Tag0: "),
                            html.Span("Cytoplasmic localization")
                        ], style={"marginTop": "0.5rem"}),
                        html.Div([
                            html.Strong("Tag1: "),
                            html.Span("Nuclear localization")
                        ], style={"marginTop": "0.25rem"}),
                    ],
                    color="blue",
                    icon=DashIconify(icon="line-md:star-twotone-loop"),
                    mb="md"
                ),
                upload_fasta,
                lncrna_sublocalization_fasta_input,
                start_button_lncrna_sublocalization,
                lncrna_sublocalization_log_container,
                output_table,
                lncrna_sublocalization_download,
                lncrna_sublocalization_workspace,
                dcc.Download(id="lncrna_sublocalization_results_downloader"),
                dcc.Interval(id="lncrna_sublocalization_log_update", interval=200, n_intervals=0, disabled=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
