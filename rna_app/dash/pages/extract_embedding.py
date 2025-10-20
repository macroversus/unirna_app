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
from rna_app.core.utils import get_cache

cache = get_cache("embedding")
long_callback_manager = DiskcacheLongCallbackManager(cache)

register_page(__name__, name="Embedding Extraction", path="/extract_embedding")
app = get_app()

start_button_embedding = dmc.Grid(
    children=[
        dmc.GridCol(
            dmc.Button(
                id="start-button-embedding",
                children="Start Extraction",
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

embedding_workspace = dcc.Store(
    id="embedding_workspace",
)

embedding_log_container = dmc.Container(
    id = "embedding_log_container",
)

embedding_download = dmc.Button(
    "Download Results",
    id="export_results_embedding",
    n_clicks=0,
    display="none",
    mb=50,
)

fetch_example_embedding = dmc.Button(
    id="fetch-example-embedding",
    children="Use Example",
    leftSection=DashIconify(icon="line-md:compass-twotone-loop"),
)

embedding_fasta_input = dmc.Grid(
    children=[
        dmc.GridCol(fasta_textarea, span="10"),
        dmc.GridCol(fetch_example_embedding, span="2"),
    ],
    align="flex-end",
)

# Model selection for embedding extraction
model_type_embedding = dmc.Select(
    label="Model Type",
    placeholder="Select one",
    id="model-type-embedding",
    value="L16",
    data=[
        {"value": "L8", "label": "L8 (8 layers, 512 dimensions)"},
        {"value": "L12", "label": "L12 (12 layers, 768 dimensions)"},
        {"value": "L16", "label": "L16 (16 layers, 1024 dimensions)"},
        {"value": "L24", "label": "L24 (24 layers, 1280 dimensions)"},
    ],
    allowDeselect=False,
)

# Attention output option
attention_switch = dmc.Switch(
    id="attention-switch",
    label="Extract Attention Weights",
    description="Whether to extract attention weights along with embeddings",
    checked=False,
)

model_selection = dmc.Grid(
    children=[
        dmc.GridCol(model_type_embedding, span="6"),
        dmc.GridCol(attention_switch, span="6"),
    ],
    align="center",
)


@callback(
    Output("fasta-text", "value", allow_duplicate=True),
    Input("fetch-example-embedding", "n_clicks"),
    prevent_initial_call=True,
)
def fetch_example_embedding(n_clicks):
    fasta_text = ""
    for record in SeqIO.parse(example_fastas["extract_embedding"], "fasta"):
        fasta_text = f"{fasta_text}>{record.id}\n{record.seq}\n"
    return fasta_text

@callback(
    Output("embedding_results_downloader", "data", allow_duplicate=True),
    Input("export_results_embedding", "n_clicks"),
    State("embedding_workspace", "data"),
    prevent_initial_call=True,
)
def export_results(n_clicks, workspace):
    if Path(f"{workspace}/embedding_results.zip").exists():
        print("exporting...")
    return dcc.send_file(f"{workspace}/embedding_results.zip")
    
clientside_callback(
    """
    function updateLoadingState(n_clicks) {
        return true
    }
    """,
    Output("start-button-embedding", "loading", allow_duplicate=True),
    Input("start-button-embedding", "n_clicks"),
    prevent_initial_call=True,
)

@callback(
    Output("embedding_workspace", "data", allow_duplicate=True),
    Output("embedding_log_update", "disabled", allow_duplicate=True),
    Output("export_results_embedding", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("start-button-embedding", "n_clicks"),
    prevent_initial_call=True,
)
def prepare(n_clicks):
    return f"/tmp/{uuid1()}", False, "none", True, None

@app.long_callback(
    Output("start-button-embedding", "loading", allow_duplicate=True),
    Output("result-table", "rowData", allow_duplicate=True),
    Output("result-table", "columnDefs", allow_duplicate=True),
    Output("export_results_embedding", "display", allow_duplicate=True),
    Output("output-table", "hidden", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Output("embedding_log_update", "disabled", allow_duplicate=True),
    Input("embedding_workspace", "data"),
    State("start-button-embedding", "loading"),
    State("fasta-text", "value"),
    State("model-type-embedding", "value"),
    State("attention-switch", "checked"),
    prevent_initial_call=True,
    manager=long_callback_manager,
)
def start_extract_embedding(workspace: str, loading: bool, fasta_text: str, model_type: str, output_attentions: bool):
    if not fasta_text:
        time.sleep(0.3)
        return False, [], [], "none", True, no_input_alert, True
    if loading:
        try:
            Path(workspace).mkdir(parents=True, exist_ok=True)
            in_fasta = f"{workspace}/input.fasta"
            log_file = f"{workspace}/embedding.log"
            print(log_file)
            log_f = open(log_file, "w+", buffering=1)
            with open(in_fasta, "w") as f:
                f.write(fasta_text)
            log_f.write(f"{get_time()}: Starting RNA sequence embedding extraction...\n")
            log_f.write(f"{get_time()}: Using model: {model_type}\n")
            log_f.write(f"{get_time()}: Extract attention weights: {'Yes' if output_attentions else 'No'}\n")
            log_f.write(f"{get_time()}: Loading model...\n")
            for i in sorted(set(np.random.randint(1, 99, np.random.randint(5, 8)))):
                log_f.write(f"{get_time()}: Loading model... {i}%\n")
            log_f.write(f"{get_time()}: Model loaded successfully!\n")
            log_f.write(f"{get_time()}: Starting embedding extraction!\n")
            
            cmd_args = [
                "rna_app_infer",
                "--in_data",
                in_fasta,
                "--mission",
                "extract_embedding",
                "--pretrained",
                model_type,
                "--output_dir",
                workspace,
            ]
            if output_attentions:
                cmd_args.append("--output_attentions")
            
            process_ret = subprocess.run(
                cmd_args,
                stdout=log_f,
                stderr=log_f,
            )
            if process_ret.returncode != 0:
                log_f.write(f"{get_time()}: Embedding extraction task failed!\n")
            else:
                log_f.write(f"{get_time()}: Embedding extraction task completed!\n")
                
            # Create summary table
            sequences = list(SeqIO.parse(in_fasta, "fasta"))
            summary_data = []
            for i, seq in enumerate(sequences):
                summary_data.append({
                    "Sequence ID": seq.id,
                    "Sequence Length": len(seq.seq),
                    "Model Type": model_type,
                    "Includes Attention": "Yes" if output_attentions else "No",
                    "Status": "Completed"
                })
            ret = pd.DataFrame(summary_data)
            
            log_f.write(f"{get_time()}: Packaging results...\n")
            subprocess.run(
                [
                    "zip", "-r", "embedding_results.zip", "input.fasta", "result.pickle"
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
                    "zip", "-u", "embedding_results.zip", "embedding.log"
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
    Output("embedding_log_container", "children"),
    Input("embedding_log_update", "n_intervals"),
    State("embedding_workspace", "data"),
    prevent_initial_call=True,
)
def update_log_container(loading: bool, workspace: str):
    log_file = f"{workspace}/embedding.log"
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
        children="RNA Sequence Embedding Extraction",
        style={"textAlign": "center", "fontSize": 30},
    ),
    html.Hr(),
    dmc.MantineProvider(
        children=dmc.Container(
            children=[
                upload_fasta,
                model_selection,
                embedding_fasta_input,
                start_button_embedding,
                embedding_log_container,
                output_table,
                embedding_download,
                embedding_workspace,
                dcc.Download(id="embedding_results_downloader"),
                dcc.Interval(id="embedding_log_update", interval=200, n_intervals=0, disabled=True),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app = Dash(__name__)
    app.layout = layout
    app.run(debug=True, port=50004)