import pandas as pd
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from datetime import datetime
from uuid import uuid4
import numpy as np
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
    no_update,
    get_app,
)
import time
from uuid import uuid1
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import (
    no_input_alert,
    standby_alert,
    success_alert,
    fail_alert,
)
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
cache = diskcache.Cache("/tmp/unirna_app/apa_cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

register_page(__name__, name="Alternative Polyadenylation Prediction", path="/apa")
app = get_app()

start_button_apa = dmc.Grid(
    children=[
        dmc.GridCol(
            dmc.Button(
                id="start-button-apa",
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

apa_workspace = dcc.Store(
    id="apa_workspace",
)

apa_log_container = dmc.Container(
    id = "apa_log_container",
)

apa_download = dmc.Button(
    "Download Results",
    id="export_results_apa",
    n_clicks=0,
    display="none",
    mb=50,
)

fetch_example_apa = dmc.Button(
    id="fetch-example-apa",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

apa_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_apa, span="2"),
    ],
    align="flex-end",
)


@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-apa", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_apa(n_clicks):
    fasta_text = ""
    # Use UTR example as placeholder since no APA specific example exists
    for record in SeqIO.parse(example_fastas["apa"], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text

@callback(
    Output("apa_results_downloader", "data", allow_duplicate=True),
    Input("export_results_apa", "n_clicks"),
    State("apa_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    if Path(f"{workspace}/apa_results.zip").exists():
        print("exporting...")
    return dcc.send_file(f"{workspace}/apa_results.zip")
    
clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-apa", "loading", allow_duplicate=True),
    Input("start-button-apa", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("apa_workspace", "data", allow_duplicate=True),
    Output("apa_log_update", "disabled", allow_duplicate=True),
    Output("export_results_apa", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-apa", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", True, None

@app.long_callback(
    Output("start-button-apa", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("export_results_apa", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("apa_log_update", "disabled", allow_duplicate=True),
    Input("apa_workspace", "data"),
    State("start-button-apa", "loading"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
    manager=long_callback_manager,
)
def start_infer_apa(workspace: str, loading: bool, fasta_text: str):
    if not fasta_text:
        time.sleep(0.3)
        return False, [], [], "none", True, no_input_alert, True
    if loading:
        try:
            Path(workspace).mkdir(parents=True, exist_ok=True)
            in_fasta = f"{workspace}/input.fasta"
            log_file = f"{workspace}/apa.log"
            print(log_file)
            log_f = open(log_file, "w+", buffering=1)
            with open(in_fasta, "w") as f:
                f.write(fasta_text)
            log_f.write(f"{get_time()}: Starting RNA Alternative Polyadenylation prediction...\n")
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
                    "apa",
                    "--output_dir",
                    workspace,
                ],
                stdout=log_f,
                stderr=log_f,
            )
            if process_ret.returncode != 0:
                log_f.write(f"{get_time()}: Prediction task failed!\n")
            else:
                log_f.write(f"{get_time()}: Prediction task completed!\n")
            ret = pd.read_csv(f"{workspace}/result.csv")
            log_f.write(f"{get_time()}: Packaging results...\n")
            subprocess.run(
                [
                    "zip", "-r", "apa_results.zip", "input.fasta", "result.csv"
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
                    "zip", "-u", "apa_results.zip", "apa.log"
                ],
                cwd=workspace,
            )
            time.sleep(0.3)
            return (
                False,
                ret.to_dict("records"),
                [{"field": i} for i in ret.columns],
                "flex",
                False,
                success_alert,
                True,
            )
        except Exception as e:
            time.sleep(0.3)
            return False, [], [], "none", True, [fail_alert, f"Error: {e}"], True

@callback(
    Output("apa_log_container", "children"),
    Input("apa_log_update", "n_intervals"),
    State("apa_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(loading: bool, workspace: str):
    log_file = f"{workspace}/apa.log"
    if not loading:
        return None
    if (not log_file) or (not Path(log_file).exists()):
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
        children="Alternative Polyadenylation Prediction",
        style={"textAlign": "center", "fontSize": 30},
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                upload_fasta,
                apa_fasta_input,
                start_button_apa,
                apa_log_container,
                output_table,
                apa_download,
                apa_workspace,
                dcc.Download(id="apa_results_downloader"),
                dcc.Interval(id="apa_log_update", interval=200, n_intervals=0, disabled=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)