import pandas as pd
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from datetime import datetime
from uuid import uuid4
import numpy as np
from tempfile import TemporaryDirectory
import subprocess
import base64
import os
from pathlib import Path
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
cache = diskcache.Cache("/tmp/unirna_app/drugrank_cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

register_page(__name__, name="RNA-Drug Interaction Ranking", path="/drugrank")
app = get_app()

start_button_drugrank = dmc.Grid(
    children=[
        dmc.GridCol(
            dmc.Button(
                id="start-button-drugrank",
                children="Start Prediction",
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

drugrank_workspace = dcc.Store(
    id="drugrank_workspace",
)

drugrank_log_container = dmc.Container(
    id = "drugrank_log_container",
)

drugrank_download = dmc.Button(
    "Download Results",
    id="export_results_drugrank",
    n_clicks=0,
    display="none",
    mb=50,
)

# CSV Upload Section
csv_upload = dcc.Upload(
    id="upload-csv-drugrank",
    children=[
        dmc.Text("Drag and Drop or", style={"display": "inline-block"}),
        dmc.Button(
            "Select CSV File",
            radius="md",
            style={
                "display": "inline-block",
                "borderWidth": "0px",
                "fullWidth": True,
            },
            leftSection=DashIconify(icon="line-md:upload-outline-loop"),
        ),
        dmc.Text("Must contain 'smiles_prepared' column"),
    ],
    max_size=11485760,
    multiple=False,
    style={
        "lineHeight": "60px",
        "borderWidth": "1px",
        "borderStyle": "dashed",
        "borderRadius": "5px",
        "textAlign": "center",
        "marginBottom": "10px",
    },
)

csv_textarea = dmc.Textarea(
    id="csv-text-drugrank",
    label="Or paste CSV content here",
    placeholder="smiles_prepared,other_columns...\nCCc1ccc(cc1)-n1nnc...,value1\n...",
    autosize=True,
    maxRows=8,
)

fetch_example_csv_drugrank = dmc.Button(
    id="fetch-example-csv-drugrank",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

drugrank_csv_input = dmc.Grid(
    children=[
        dmc.GridCol(csv_textarea, span="10"),
        dmc.GridCol(fetch_example_csv_drugrank, span="2"),
    ],
    align="flex-end",
)

# FASTA input (reuse from existing)
fetch_example_drugrank = dmc.Button(
    id="fetch-example-drugrank",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

drugrank_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_drugrank, span="2"),
    ],
    align="flex-end",
)

# CSV input section
csv_input_section = dmc.Container([
    dmc.Title("Step 1: Upload SMILES Data (CSV)", order=4, mb="sm"),
    csv_upload,
    drugrank_csv_input,
], mb="md")

# FASTA input section  
fasta_input_section = dmc.Container([
    dmc.Title("Step 2: Upload RNA Sequences (FASTA)", order=4, mb="sm"),
    upload_fasta,
    drugrank_fasta_input,
], mb="md")


@callback(
    Output("csv-text-drugrank", "value", allow_duplicate=True),
    Output("upload-csv-drugrank", "contents"),
    Input("upload-csv-drugrank", "contents"),
    Input("upload-csv-drugrank", "last_modified"),
    prevent_initial_call=True,
)
def load_csv_file(contents, last_modified):
    if not contents:
        return "", None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # Try to decode as CSV
        csv_text = decoded.decode('utf-8')
        return csv_text, None
    except:
        return "Error: Could not decode CSV file", None


@callback(
    Output("csv-text-drugrank", "value", allow_duplicate=True),
    Input("fetch-example-csv-drugrank", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_csv_drugrank(n_clicks):
    # Use fse_vs_chemdiv.csv from example as CSV example
    example_csv = "/home/dingsz/project/rna_app_review/unirna_app/example/drugrank/fse_vs_chemdiv.csv"
    try:
        with open(example_csv, 'r') as f:
            csv_text = f.read()
        return csv_text
    except:
        # Fallback example
        csv_text = "smiles_prepared,plants_score,ledock_score\nCCc1ccc(cc1)-n1nnc(C(=O)NCc2ccncc2)c1-c1cccnc1,-10.404,-7.98\nCOc1ccc(cc1)-c1cc(C(=O)N(C)Cc2ccn[nH]2)c2c(C)nn(C)c2n1,-18.4964,-7.48\n"
        return csv_text


@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-drugrank", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_drugrank(n_clicks):
    # Use test.fasta from drugrank example as example
    fasta_text = ""
    example_fasta = "/home/dingsz/project/rna_app_review/unirna_app/example/drugrank/test.fasta"
    try:
        for record in SeqIO.parse(example_fasta, "fasta"):
            fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    except:
        # Fallback example
        fasta_text = ">hsa-let-7f-1\nUCAGAGUGAGGUAGUAGAUUGUAUAGUUGUGGGGUAGUGAUUUUACCCUGUUCAGGAGAUAACUAUACAA\n>hsa-mir-31\nGGAGAGGAGGCAAGAUGCUGGCAUAGCUGUUGAACUGGGAACCUGCUAUGCCAACAAUAUUGCCAUCUUUCC\n"
    return fasta_text


@callback(
    Output("drugrank_results_downloader", "data", allow_duplicate=True),
    Input("export_results_drugrank", "n_clicks"),
    State("drugrank_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    if Path(f"{workspace}/drugrank_results.zip").exists():
        print("exporting...")
    return dcc.send_file(f"{workspace}/drugrank_results.zip")
    

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-drugrank", "loading", allow_duplicate=True),
    Input("start-button-drugrank", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("drugrank_workspace", "data", allow_duplicate=True),
    Output("drugrank_log_update", "disabled", allow_duplicate=True),
    Output("export_results_drugrank", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-drugrank", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", True, None


@app.long_callback(
    Output("start-button-drugrank", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("export_results_drugrank", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("drugrank_log_update", "disabled", allow_duplicate=True),
    Input("drugrank_workspace", "data"),
    State("start-button-drugrank", "loading"),
    State("csv-text-drugrank", "value"),
    State("fasta-text", "value"),
    prevent_initial_call=True,
    manager=long_callback_manager,
)
def start_infer_drugrank(workspace: str, loading: bool, csv_text: str, fasta_text: str):
    if not csv_text or not fasta_text:
        time.sleep(0.3)
        return False, [], [], "none", True, no_input_alert, True
    if loading:
        try:
            Path(workspace).mkdir(parents=True, exist_ok=True)
            
            # Save CSV file
            csv_path = f"{workspace}/input_smiles.csv"
            with open(csv_path, "w") as f:
                f.write(csv_text)
            
            # Save FASTA file
            fasta_path = f"{workspace}/input_sequences.fasta"
            with open(fasta_path, "w") as f:
                f.write(fasta_text)
            
            log_file = f"{workspace}/drugrank.log"
            print(log_file)
            log_f = open(log_file, "w+", buffering=1)
            
            log_f.write(f"{get_time()}: 准备开始RNA-药物相互作用排名预测...\n")
            log_f.write(f"{get_time()}: 模型载入中...\n")
            for i in sorted(set(np.random.randint(1, 99, np.random.randint(5, 8)))):
                log_f.write(f"{get_time()}: 模型载入中... {i}%\n")
            log_f.write(f"{get_time()}: 模型载入完成！\n")
            log_f.write(f"{get_time()}: 开始预测！\n")
            
            process_ret = subprocess.run(
                [
                    "rna_app_infer",
                    "--in_data",
                    csv_path,  # DrugRank uses CSV as primary input
                    "--mission",
                    "drugrank",
                    "--output_dir",
                    workspace,
                    # Additional arguments could be added here for FASTA path
                ],
                stdout=log_f,
                stderr=log_f,
                env={**os.environ, "DRUGRANK_FASTA_PATH": fasta_path}  # Pass FASTA path via env
            )
            if process_ret.returncode != 0:
                log_f.write(f"{get_time()}: 预测任务发生错误！\n")
            else:
                log_f.write(f"{get_time()}: 预测任务完成！\n")
            ret = pd.read_csv(f"{workspace}/result.csv")
            log_f.write(f"{get_time()}: 打包结果中...\n")
            subprocess.run(
                [
                    "zip", "-r", "drugrank_results.zip", "input_smiles.csv", "input_sequences.fasta", "result.csv"
                ],
                cwd=workspace,
                stdout=log_f,
                stderr=log_f,
            )
            log_f.write(f"{get_time()}: 结果打包完成！\n")
            log_f.flush()
            log_f.close()
            subprocess.run(
                [
                    "zip", "-u", "drugrank_results.zip", "drugrank.log"
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
    Output("drugrank_log_container", "children"),
    Input("drugrank_log_update", "n_intervals"),
    State("drugrank_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(loading: bool, workspace: str):
    log_file = f"{workspace}/drugrank.log"
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
        children="RNA-Drug Interaction Ranking Prediction",
        style={"textAlign": "center", "fontSize": 30},
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                dmc.Alert(
                    title="DrugRank Multi-Modal Prediction",
                    children="This tool predicts binding affinity rankings between RNA sequences and small molecule compounds using UniMol (molecular features) and UniRNA (RNA features) with XGBoost ranking.",
                    color="blue",
                    icon=DashIconify(icon="line-md:star-twotone-loop"),
                    mb="md"
                ),
                csv_input_section,
                fasta_input_section,
                start_button_drugrank,
                drugrank_log_container,
                output_table,
                drugrank_download,
                drugrank_workspace,
                dcc.Download(id="drugrank_results_downloader"),
                dcc.Interval(id="drugrank_log_update", interval=200, n_intervals=0, disabled=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)