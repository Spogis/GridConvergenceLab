import dash
from dash import dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

from dash import html
from dash import dcc
from dash import dash_table


def layout_GCI_First_Analysis():
    layout = html.Div([

        html.Div([
            dash_table.DataTable(
                id='mesh-phi-table',
                columns=[
                    {'name': 'Mesh', 'id': 'Mesh', 'type': 'numeric'},
                    {'name': 'phi', 'id': 'phi', 'type': 'numeric'},
                ],
                data=[],
                editable=True,
                row_deletable=False,
                style_table={'width': '80%', 'margin': '0 auto'},
                style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial, sans-serif'},
                style_header={'backgroundColor': '#3498DB', 'color': 'white', 'fontWeight': 'bold'}
            ),
        ], style={'marginBottom': '20px'}),

        html.Div([
            dcc.Upload(
                id='upload-phi-table',
                children=html.Div([
                    'Drag and drop or ',
                    html.A('select data file', style={'color': '#1ABC9C', 'fontWeight': 'bold'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#ECF0F1'
                }
            ),
            html.Div(id='output-phi-table', style={'textAlign': 'center', 'marginTop': '10px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'flexWrap': 'wrap'}),

        dcc.Graph(id='output-graph_mesh', style={'margin': '20px', 'display': 'none'})

    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})

    return layout
