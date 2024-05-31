import pandas as pd
from Bio import SeqIO
from datetime import datetime
import io
import dash_mantine_components as dmc
from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, register_page
from rna_app.dash.pages.utils import *
from rna_app.dash.pages.alerts import no_input_alert, standby_alert, success_alert
from rna_app.core.acceptor import infer_acceptor
from rna_app.core.donor import infer_donor

# register_page(__name__)


acceptor_or_donar = dmc.RadioGroup(
    id="acceptor-or-donar",
    children=[
        "Select the type of splice site you want to infer:",
        dmc.Group(
            [
                dmc.Radio(label="Acceptor", value="acceptor"),
                dmc.Radio(label="Donar", value="donar"),
            ],
        ),
    ],
    value="acceptor",
    style={"display": "flex", "margin": "20px"},
) 

@callback(
    Output("start-button", "loading", allow_duplicate=True),
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Output("result-table", "export_format"),
    Output("status", "children"),
    Input("fasta-text", "value"),
    Input("start-button", "loading"),
    Input("acceptor-or-donar", "value"),
    prevent_initial_call=True,
)
def set_start_button_status(fasta_text: str, loading: bool, splice_type: str):
    print(datetime.now())
    if not fasta_text:
        return False, [], [], None, no_input_alert
    if fasta_text and loading:
        try:
            parser = SeqIO.parse(io.StringIO(fasta_text), "fasta")
            output_dir = OUTPUT_ROOT / splice_type
            if splice_type == "acceptor":
                ret: pd.DataFrame = infer_acceptor(parser, output_dir, return_df=True) 
            elif splice_type == "donar":
                ret: pd.DataFrame = infer_donor(parser, output_dir, return_df=True)
            else:
                raise ValueError("Invalid splice type")
            return False, ret.to_dict("records"), [{"name": i, "id": i} for i in ret.columns], "csv", success_alert
        except Exception as e:
            return False, [], [], None, dmc.Alert(
                children=f"Error: {e}. Please check your input. It should be in FASTA format. If you are sure that your input is correct, please contact the developer.",
                title="Error",
                color="red",
            )
    else:
        return False, [], [], None, standby_alert

app = Dash(__name__)

app.layout = [
    html.Div(children="Splice Site", style={"textAlign": "center", "fontSize": 30}),
    html.Hr(),
    dmc.MantineProvider(
        children = dmc.Container(
            children = [
                upload_fasta,
                fasta_textarea,
                dmc.MantineProvider(acceptor_or_donar),
                start_button,
                status,
                html.Div(
                    children=output_table,
                    style={
                        "margin": "auto",
                    }
                ),
            ],
        ),
    ),
]

if __name__ == "__main__":
    app.run(debug=True, port=50004)
