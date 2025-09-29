import os
os.environ["REACT_VERSION"] = "18.2.0"
import dash
from dash import Dash, html, dcc, callback, Input, Output
from dash_iconify import DashIconify
import dash_mantine_components as dmc
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
        dmc.Divider(label="Molecule Simulates the Future", labelPosition="center", variant="dashed", size="xs", color="blue"),
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
                         dmc.Anchor("https://www.biosino.org/rdbsb/", href="https://www.biosino.org/rdbsb/", target="_blank", c="blue", style={"textDecoration": "underline"}),
                         ") under the accession numbers OENR1-OENR11903. Model weights can be found at ",
                         dmc.Anchor("https://github.com/macroversus/unirna_app", href="https://github.com/macroversus/unirna_app", target="_blank", c="blue", style={"textDecoration": "underline"}),
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
                     "Institute for Advanced Algorithms Research, Shanghai. (hereinafter referred to as “IAAR”) fully possesses the copyright of this code. Without IAAR’s written authorization, any natural person or enterprise shall not copy, forward or perform any unauthorized act. If authorized, the source must be clearly indicated. Otherwise, IAAR has the right to pursue liability for the violation of the above mentioned terms, and reserves the right to pursue other legal responsibilities.",
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
    app.run_server(debug=False, port=50004, host="0.0.0.0")
