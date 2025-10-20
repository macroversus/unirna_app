import os
os.environ["REACT_VERSION"] = "18.2.0"
import dash
from dash import Dash, html, dcc, callback, Input, Output
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import argparse
from pathlib import Path

app = Dash(__name__, use_pages=True, pages_folder="pages")
server = app.server

navbar = dmc.Container(
    children=[
        dmc.Anchor(
            href="/",
            children=[
                    dmc.Button(
                        "Home",
                        fw=700,
                        size="md",
                    ),
                ],
            ),
        dmc.Group(
            children=[
                dmc.Anchor(dmc.Button("Explore Apps"), href="/apps", size="md"),
                dmc.Anchor(dmc.Button("Contact"), href="/contact", size="md"),
            ],
            style={"marginLeft": "auto"},
        ),        
    ],
    style={
        "marginTop": "2%",  
    },
    display="flex",
    size="xl",
)

app.layout = dmc.MantineProvider(
    children=[
        navbar,
        dmc.Divider(variant="dashed", size="xs", color="blue"),
        html.H1("Uni-RNA Apps", style={"textAlign": "center"}),
        
        # Uni-RNA paper description section
        dmc.Container(
            children=[
                 dmc.Text(
                     [
                         "This website contains the implementations for the main downstream tasks mentioned in the Uni-RNA paper. ",
                         "Especially, as the tRNA section is patent-pending, its corresponding raw data is provided exclusively on this site and should not be used for any other purpose (",
                         dmc.Button(
                             "Click here to access the data",
                             id="btn-download-trna",
                             variant="transparent",
                             size="compact-sm",
                             c="blue",
                             style={
                                 "textDecoration": "underline", 
                                 "padding": "0", 
                                 "height": "auto", 
                                 "fontWeight": "normal",
                                 "backgroundColor": "transparent",
                                 "border": "none",
                                 "cursor": "pointer",
                                 "display": "inline",
                                 "fontSize": "inherit",
                                 "lineHeight": "inherit"
                             },
                             styles={
                                 "root": {
                                     "&:hover": {
                                         "backgroundColor": "transparent !important",
                                         "color": "#1c7ed6 !important",
                                         "textDecoration": "underline"
                                     }
                                 }
                             }
                         ),
                         "). Other experimental results and raw data are available at the Registry and database of bioparts for synthetic biology (",
                         html.A("https://www.biosino.org/rdbsb/", href="https://www.biosino.org/rdbsb/", target="_blank", style={"color": "blue", "textDecoration": "underline"}),
                         ") under the accession numbers OENR1-OENR11903. Model weights can be found at ",
                         html.A("https://github.com/macroversus/unirna_app", href="https://github.com/macroversus/unirna_app", target="_blank", style={"color": "blue", "textDecoration": "underline"}),
                         ", and the wget download password is ",
                         dmc.Code("LzzM5OQtTGKYHSwpQqOAn6fL7Lu1medN", style={"backgroundColor": "#f8f9fa", "padding": "2px 4px", "borderRadius": "3px"}),
                         "."
                     ],
                     size="sm",
                     style={
                         "textAlign": "justify",
                         "lineHeight": "1.6",
                         "marginBottom": "1rem",
                         "color": "#495057"
                     }
                 ),
                 dmc.Alert(
                     children=[
                         dmc.Group([
                             dmc.Badge("NEW: 2025-10-20", color="green", size="sm", variant="filled"),
                             dmc.Text(
                                 "New parameters have been added to the sequence optimization tool and the weight for tRNA sequence optimization has been updated based on new experimental results.",
                                 size="sm"
                             ),
                         ], align="center", gap="xs"),
                     ],
                     color="green",
                     variant="light",
                     mb="md",
                 ),
                 dmc.Divider(
                     label="© Copyright Notice",
                     labelPosition="left",
                     variant="solid",
                     size="xs",
                     color="#e9ecef",
                     style={"marginTop": "1rem", "marginBottom": "1rem"},
                     styles={
                         "label": {
                             "fontSize": "1.1rem",
                             "fontWeight": "bold",
                             "color": "#495057"
                         }
                     }
                 ),
                 dmc.Text(
                     "The Shanghai Institute for Advanced Algorithms Research (hereinafter referred to as “IAAR”) and Beijing DP Technology (hereinafter referred to as “DP Technology”) fully possess the copyright to this code. Without the written authorization of IAAR and DP Technology, no natural person or enterprise shall copy, forward, or perform any unauthorized act. If authorization is granted, the source must be clearly indicated. Otherwise, IAAR and DP Technology have the right to hold the violator liable for violating the above-mentioned terms and reserve the right to pursue further legal action.",
                     size="sm",
                     style={
                         "textAlign": "justify",
                         "lineHeight": "1.6",
                         "marginBottom": "1.5rem",
                         "color": "#6c757d",
                         "fontStyle": "italic"
                     }
                 )
            ],
            size="lg",
            style={
                "marginTop": "1rem",
                "marginBottom": "1rem",
                "padding": "0 2rem"
            }
        ),
        
        # Download component for tRNA data
        dcc.Download(id="download-trna-data"),
        
        dash.page_container,
    ],
)

# Callback for tRNA data download
@callback(
    Output("download-trna-data", "data"),
    Input("btn-download-trna", "n_clicks"),
    prevent_initial_call=True,
)
def download_trna_data(n_clicks):
    """Handle tRNA data download when button is clicked"""
    if n_clicks:
        # 获取tRNA数据文件路径
        file_path = Path(__file__).parent / "assets" / "tRNA_meta_data.xlsx"
        
        # 使用dcc.send_file发送文件
        return dcc.send_file(str(file_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Uni-RNA Dash App")
    parser.add_argument("--debug", action="store_true", default=False, help="Run in debug mode")
    parser.add_argument("--port", type=int, default=50004, help="Port to run the server on (default: 50004)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on (default: 0.0.0.0)")
    args = parser.parse_args()
    
    app.run_server(debug=args.debug, port=args.port, host=args.host)
