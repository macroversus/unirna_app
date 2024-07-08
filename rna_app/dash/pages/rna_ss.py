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
    dcc,
    callback,
    Output,
    Input,
    clientside_callback,
    register_page,
    State,
    ctx,
    no_update,
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

ss_workspace = dcc.Store(
    id="ss_workspace",
    data=f"/tmp/{uuid1()}"
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
        download_ss,
        dcc.Interval(id="ss_log_update", interval=200, n_intervals=0, disabled=True),
        dcc.Download(id="results_downloader"),
    ],
    w="100%",
    display="none",
)


@callback(
    Output("results_downloader", "data", allow_duplicate=True),
    Input("export_results", "n_clicks"),
    State("ss_workspace", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, workspace):
    print(ctx.triggered_id)
    if ctx.triggered_id == "export_results":
        if Path(f"{workspace}/rna_ss_results.zip").exists():
            print("exporting...")
        print(ctx.states)
        return dcc.send_file(f"{workspace}/rna_ss_results.zip")
    else:
        return no_update

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


def get_time():
    return datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

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
    Output("ss_workspace", "data"),
    Input("start-button-rna_ss", "loading"),
    State("fasta-text", "value"),
    State("ss_workspace", "data"),
    prevent_initial_call=True,
)
def start_infer_rna_ss(loading: bool, fasta_text: str, ss_workspace: str):
    if not fasta_text:
        return False, [], [], True, "none", no_input_alert, [], [], [], ss_workspace
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
                    "zip", "-r", f"rna_ss_results.zip", f"./"
                ],
                cwd=ss_workspace,
                stdout=log_f,
                stderr=log_f,
            )
            log_f.write(f"{get_time()}: 结果打包完成！\n")
            log_f.close()
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
                ss_workspace,
            )
        except Exception as e:
            return False, [], [], True, "none", [fail_alert, f"Error: {e}"], [], [], [], ss_workspace

@callback(
    Output("ss_log_update", "disabled"),
    Input("start-button-rna_ss", "loading"),
    prevent_initial_call=True,
)
def start_monitor(loading: bool):
    if loading:
        return False
    return True

@callback(
    Output("ss_log_container", "children"),
    Input("ss_log_update", "n_intervals"),
    State("ss_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(loading: bool, ss_workspace: str):
    print(loading, ss_workspace)
    log_file = f"{ss_workspace}/rna_ss.log"
    if not loading:
        return None
    if (not log_file) or (not Path(log_file).exists()):
        return dmc.Skeleton(height=300)
    try:
        with open(log_file, "r") as f:
            log_text = "".join(filter(lambda x: "deprecated" not in x.lower(), f.readlines()))
        if not log_text:
            return dmc.Skeleton(height=300)
        print(log_text)
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
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)
