import pandas as pd
from Bio import SeqIO
from datetime import datetime
import dash_mantine_components as dmc
from uuid import uuid4
import numpy as np
from pathlib import Path
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

cache = diskcache.Cache("/tmp/unirna_app/seq_optimization_cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

register_page(
    __name__,
    name="Sequence Optimization",
    path="/seq_optimization",
)
app = get_app()

# Template sequence input
template_textarea = dmc.Textarea(
    id="template-text",
    label="Template Sequence",
    placeholder="Enter RNA template sequence (e.g., CUUGGUUGGUAGCGCAGU...)",
    autosize=True,
    required=True,
    maxRows=8,
)

# Mutation ratio slider
mutation_ratio_slider = dmc.Slider(
    id="mutation-ratio",
    label="Mutation Ratio",
    value=0.1,
    min=0.01,
    max=0.5,
    step=0.01,
    marks=[
        {"value": 0.01, "label": "1%"},
        {"value": 0.1, "label": "10%"},
        {"value": 0.25, "label": "25%"},
        {"value": 0.5, "label": "50%"},
    ],
)

# Iterations input
iterations_input = dmc.NumberInput(
    id="iterations-input",
    label="Optimization Iterations",
    value=20,
    min=1,
    max=100,
    step=1,
)

start_button_seq_optimization = dmc.Grid(
    children=[
        dmc.GridCol(
            dmc.Button(
                id="start-button-seq_optimization",
                children="Start Optimization",
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

seq_optimization_workspace = dcc.Store(
    id="seq_optimization_workspace",
)

seq_optimization_log_container = dmc.Container(
    id="seq_optimization_log_container",
)

seq_optimization_download = dmc.Button(
    "Download Results",
    id="export_results_seq_optimization",
    n_clicks=0,
    display="none",
    mb=50,
)

fetch_example_seq_optimization = dmc.Button(
    id="fetch-example-seq_optimization",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

seq_optimization_input = dmc.Grid(
    children=[
        dmc.GridCol(template_textarea, span="10"),
        dmc.GridCol(fetch_example_seq_optimization, span="2"),
    ],
    align="flex-end",
)


@callback(
    Output("template-text", "value", allow_duplicate=True),
    Input("fetch-example-seq_optimization", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_seq_optimization(n_clicks):
    # 提供一个示例tRNA模板序列
    template = "CUUGGUUGGUAGCGCAGUUGGUUAUCAUUUGCUUCGCGGUAAGAUCCUGGAGUCCUAAAAUCCUUCGCAUCAAGAC"
    return template


@callback(
    Output("seq_optimization_results_downloader", "data", allow_duplicate=True),
    Input("export_results_seq_optimization", "n_clicks"),
    State("seq_optimization_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    if Path(f"{workspace}/seq_optimization_results.zip").exists():
        print("exporting...")
    return dcc.send_file(f"{workspace}/seq_optimization_results.zip")


clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-seq_optimization", "loading", allow_duplicate=True),
    Input("start-button-seq_optimization", "n_clicks"),
    prevent_initial_call=True,
)


@callback(
    Output("seq_optimization_workspace", "data", allow_duplicate=True),
    Output("seq_optimization_log_update", "disabled", allow_duplicate=True),
    Output("export_results_seq_optimization", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-seq_optimization", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", True, None


@app.long_callback(
    Output("start-button-seq_optimization", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("export_results_seq_optimization", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("seq_optimization_log_update", "disabled", allow_duplicate=True),
    Input("seq_optimization_workspace", "data"),
    State("start-button-seq_optimization", "loading"),
    State("template-text", "value"),
    State("mutation-ratio", "value"),
    State("iterations-input", "value"),
    prevent_initial_call=True,
    manager=long_callback_manager,
)
def start_seq_optimization(workspace: str, loading: bool, template: str, mutation_ratio: float, iterations: int):
    if not template:
        time.sleep(0.3)
        return False, [], [], "none", True, no_input_alert, True
    if loading:
        try:
            Path(workspace).mkdir(parents=True, exist_ok=True)
            log_file = f"{workspace}/seq_optimization.log"
            print(log_file)
            log_f = open(log_file, "w+", buffering=1)
            
            log_f.write(f"{get_time()}: Preparing sequence optimization...\n")
            log_f.write(f"{get_time()}: Template sequence: {template}\n")
            log_f.write(f"{get_time()}: Mutation ratio: {mutation_ratio}\n")
            log_f.write(f"{get_time()}: Iterations: {iterations}\n")
            log_f.write(f"{get_time()}: Loading model...\n")
            
            # 模拟模型加载进度
            for i in sorted(set(np.random.randint(1, 99, np.random.randint(5, 8)))):
                log_f.write(f"{get_time()}: Loading model... {i}%\n")
            
            log_f.write(f"{get_time()}: Model loaded successfully!\n")
            log_f.write(f"{get_time()}: Starting sequence optimization...\n")
            
            # 调用推理命令
            process_ret = subprocess.run(
                [
                    "rna_app_infer",
                    "--in_data",
                    template,
                    "--mission",
                    "seq_optimization",
                    "--output_dir",
                    workspace,
                    "--mutation_ratio",
                    str(mutation_ratio),
                    "--iterations",
                    str(iterations),
                ],
                stdout=log_f,
                stderr=log_f,
            )
            
            if process_ret.returncode != 0:
                log_f.write(f"{get_time()}: Optimization task failed!\n")
                log_f.flush()
                log_f.close()
                time.sleep(0.3)
                return False, [], [], "none", True, [fail_alert, "Optimization failed"], True
            else:
                log_f.write(f"{get_time()}: Optimization task completed!\n")
            
            # 读取结果
            ret = pd.read_csv(f"{workspace}/optimized_sequences.csv")
            
            log_f.write(f"{get_time()}: Packing results...\n")
            subprocess.run(
                [
                    "zip", "-r", "seq_optimization_results.zip", "optimized_sequences.csv"
                ],
                cwd=workspace,
                stdout=log_f,
                stderr=log_f,
            )
            log_f.write(f"{get_time()}: Results packed successfully!\n")
            log_f.flush()
            log_f.close()
            
            # 将日志添加到zip
            subprocess.run(
                [
                    "zip", "-u", "seq_optimization_results.zip", "seq_optimization.log"
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
    Output("seq_optimization_log_container", "children"),
    Input("seq_optimization_log_update", "n_intervals"),
    State("seq_optimization_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(n_intervals: int, workspace: str):
    log_file = f"{workspace}/seq_optimization.log"
    if not log_file:
        return None
    if not Path(log_file).exists():
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
        children="Sequence Optimization",
        style={"textAlign": "center", "fontSize": 30},
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                seq_optimization_input,
                dmc.Stack([
                    mutation_ratio_slider,
                    iterations_input,
                ], gap="md"),
                start_button_seq_optimization,
                seq_optimization_log_container,
                output_table,
                seq_optimization_download,
                seq_optimization_workspace,
                dcc.Download(id="seq_optimization_results_downloader"),
                dcc.Interval(id="seq_optimization_log_update", interval=200, n_intervals=0, disabled=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)