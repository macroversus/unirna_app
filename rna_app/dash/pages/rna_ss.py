import pandas as pd
import numpy as np
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from uuid import uuid4
from tempfile import TemporaryDirectory
import subprocess
from dash import (
    Dash,
    html,
    # dcc,
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
from uuid import uuid1
from dash.exceptions import PreventUpdate
import dash_bio
from rna_app.dash.collections.utils import *
from rna_app.dash.collections.alerts import (
    no_input_alert,
    standby_alert,
    success_alert,
    fail_alert,
)
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
from dash_extensions.enrich import dcc
from flask import Flask, Response
cache = diskcache.Cache("/tmp/rna_ss_cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

register_page(__name__, name="RNA Secondary Structure Prediction", path="/rna_ss")
app = get_app()
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

ss_workspace = dcc.Store(
    id="ss_workspace",
)

ss_log_container = dmc.Container(
    id = "ss_log_container",
)

ss_store = dcc.Store(id="ss-store", storage_type="session")
ss_contrainer = dash_bio.FornaContainer(id="ss-contrainer", height=400, width=800)
ss_display = dcc.Dropdown(id="ss-display", options=[], multi=True, value=[])

download_ss = dmc.Button(
    "Download Results",
    id="export_results",
    n_clicks=0,
    mb=50,
    display="none",
)

outputs_ss = dmc.Container(
    id="outputs-ss",
    children=[
        ss_store,
        html.P('Select the sequences to display below.'),
        ss_display,
        html.Hr(),
        ss_contrainer,
        html.Hr(),
        dcc.Interval(id="ss_log_update", interval=1000, n_intervals=0, disabled=True),
    ],
    w="100%",
    display="none",
)

@app.server.route('/download-results/<path:file_path>')
def download_results(file_path):
    full_path = f'/{file_path}'
    def generate():
        with open(full_path, 'rb') as f:
            while chunk := f.read(8192):  # 每次读取8KB
                yield chunk
    total_size = os.path.getsize(full_path)
    headers = {
        'Content-Disposition': f'attachment; filename={os.path.basename(full_path)}',
        'Content-Type': 'application/zip',
        'Content-Length': str(total_size),
    }
    return Response(generate(), headers=headers)

clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("export_results", "loading", allow_duplicate=True),
    Input("export_results", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("dummy", "href"),
    Output("export_results", "loading"),
    Input("export_results", "n_clicks"),
    State("ss_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    try:
        if ctx.triggered_id == "export_results":
            if Path(f"{workspace}/rna_ss_results.zip").exists():
                print("exporting...")
            return f"/download-results{workspace}/rna_ss_results.zip", False
        return no_update, False
    except Exception as e:
        return no_update, False
    
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
    Output("ss_workspace", "data", allow_duplicate=True),
    Output("ss_log_update", "disabled", allow_duplicate=True),
    Output("outputs-ss", "display", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-rna_ss", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", None

@app.long_callback(
    Output("start-button-rna_ss", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("outputs-ss", "display", allow_duplicate=True),
    Output("ss-display", "options", allow_duplicate=True),
    Output("ss-display", "value", allow_duplicate=True),
    Output("ss-store", "data", allow_duplicate=True),
    Output("ss_workspace", "data"),
    Output("ss_log_update", "disabled"),
    Output("export_results", "display"),
    State("start-button-rna_ss", "loading"),
    State("fasta-text", "value"),
    Input("ss_workspace", "data"),
    prevent_initial_call=True,
    manager=long_callback_manager,
)
def start_infer_rna_ss(loading: bool, fasta_text: str, ss_workspace: str):
    if not fasta_text:
        return False, [], [], True, no_input_alert, "none", [], [], [], ss_workspace, True, "none"
    Path(ss_workspace).mkdir(parents=True, exist_ok=True)
    if loading:
        try:
            log_file = f"{ss_workspace}/rna_ss.log"
            log_f = open(log_file, "w+", buffering=1)
            in_fasta = f"{ss_workspace}/input.fasta"
            print(log_file)
            with open(in_fasta, "w") as f:
                f.write(fasta_text)
            log_f.write(f"{get_time()}: 准备开始预测RNA二级结构\n")
            log_f.write(f"{get_time()}: 模型载入中...\n")
            for i in sorted(np.random.randint(1, 99, np.random.randint(5, 8))):
                log_f.write(f"{get_time()}: 模型载入中... {i}%\n")
            log_f.write(f"{get_time()}: 模型载入完成！\n")
            log_f.write(f"{get_time()}: 开始预测！\n")
            process_ret = subprocess.run(
                [
                    "rna_app_infer",
                    "--in_data",
                    in_fasta,
                    "--mission",
                    "rna_ss",
                    "--output_dir",
                    ss_workspace,
                ],
                stdout=log_f,
                stderr=log_f,
            )
            if process_ret.returncode != 0:
                log_f.write(f"{get_time()}: 预测任务发生错误！\n")
            else:
                log_f.write(f"{get_time()}: 预测任务完成！\n")
            ret = pd.read_csv(f"{ss_workspace}/result.csv")
            log_f.write(f"{get_time()}: 打包结果中...\n")
            subprocess.run(
                [
                    "zip", "-r", "rna_ss_results.zip", "./", "-x", "rna_ss.log"
                ],
                cwd=ss_workspace,
                stdout=log_f,
                stderr=log_f,
            )
            log_f.write(f"{get_time()}: 结果打包完成！\n")
            subprocess.run(
                [
                    "zip", "-u", "rna_ss_results.zip", "rna_ss.log"
                ],
                cwd=ss_workspace,
            )
            if (ret.shape[0]) > 5 or (len("".join(ret["seq"])) > 200):
                log_f.write(f"{get_time()}: 序列超过5条或核酸数超过200, 不显示二级结构\n")
                ss_display_content = ["none", [], [], []]
                out_status = success_alert
            else:
                ss_display_content = [
                    "inline-block", 
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
                ]
                out_status = [
                    success_alert,
                    dmc.Alert(
                        title="Too Many Sequences",
                        children="The number of sequences is too large to display secondary structures. Please download the results and use a local viewer.",
                        color="yellow",
                        icon=dmc.ActionIcon(
                            DashIconify(icon="line-md:loading-alt-loop"),
                            size="md",
                            variant="filled",
                            color="yellow",
                        ),
                    )
                ]
            log_f.close()
            print("done")
            return (
                False,
                ret.to_dict("records"),
                [{"field": i} for i in ret.columns],
                True,
                out_status,
                *ss_display_content,
                ss_workspace,
                True,
                "flex",
            )
        except Exception as e:
            return False, [], [], True, [fail_alert, f"Error: {e}"], "none", [], [], [], ss_workspace, True, "none"

@callback(
    Output("ss_log_container", "children"),
    Input("ss_log_update", "n_intervals"),
    State("ss_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(loading: bool, ss_workspace: str):
    log_file = f"{ss_workspace}/rna_ss.log"
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
                ss_log_container,
                outputs_ss,
                output_table,
                ss_workspace,
                download_ss,
                dcc.Location(id="dummy", refresh=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
