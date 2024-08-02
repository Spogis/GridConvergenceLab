import dash
from dash import dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

from dash import html
from dash import dcc
from dash import dash_table


def load_data():
    df = pd.read_excel('setups/Var_Table.xlsx', sheet_name='Variables')
    return df.to_dict('records')


def load_mesh_sizes():
    df = pd.read_excel('setups/Var_Table.xlsx', sheet_name='MeshSizes')
    return df.to_dict('records')


def load_volume():
    df = pd.read_excel('setups/Var_Table.xlsx', sheet_name='Volume')
    print(df['volume'][0])
    return df['volume'][0]


def layout_GCI():
    layout = html.Div([
        html.Br(),
        html.H1("GCI Calculation (Grid Convergence Index)", style={
            'textAlign': 'center',
            'color': '#2C3E50',
            'fontFamily': 'Arial, sans-serif',
            'fontWeight': 'bold'
        }),

        html.Div([
            html.Label("Domain Volume:", style={
                'fontSize': '16px',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            dcc.Input(id='domain-volume', type='number', style={
                'width': '200px',
                'padding': '10px',
                'fontSize': '16px',
                'borderRadius': '10px',
                'border': '2px solid #3498DB',
                'textAlign': 'center',
                'marginBottom': '20px',
                'marginRight': '10px',
                'marginLeft': '10px'
            }),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'center',
                  'alignItems': 'center', 'gap': '10px'}),

        html.Div([
            html.Br(),
            html.Label("Mesh Type:", style={
                'fontSize': '16px',
                'fontWeight': 'bold',
                'marginBottom': '10px',
                'marginRight': '10px',
                'marginLeft': '10px',
            }),
            dcc.RadioItems(
                id='mesh-type',
                options=[
                    {'label': '3D', 'value': '3D'},
                    {'label': '2D', 'value': '2D'},
                ],
                labelStyle={
                    'width': '100px',
                    'padding': '10px',
                    'display': 'inline-block',
                    'marginRight': '20px',
                    'fontSize': '20px',
                    'borderRadius': '10px',
                    'border': '2px solid #3498DB',
                    'textAlign': 'center',
                },
                value='3D'
            ),
        ], style={'textAlign': 'center', 'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'center',
                  'alignItems': 'center', 'gap': '10px'}),

        html.Div([
            dash_table.DataTable(
                id='mesh-sizes-table',
                columns=[
                    {'name': 'Coarse Mesh', 'id': 'coarse', 'type': 'numeric'},
                    {'name': 'Medium Mesh', 'id': 'medium', 'type': 'numeric'},
                    {'name': 'Fine Mesh', 'id': 'fine', 'type': 'numeric'}
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
            dash_table.DataTable(
                id='editable-table',
                columns=[
                    {'name': 'Variable', 'id': 'variable', 'type': 'text'},
                    {'name': 'Coarse Mesh', 'id': 'coarse', 'type': 'numeric'},
                    {'name': 'Medium Mesh', 'id': 'medium', 'type': 'numeric'},
                    {'name': 'Fine Mesh', 'id': 'fine', 'type': 'numeric'}
                ],
                data=[],
                editable=True,
                row_deletable=True,
                style_table={'width': '80%', 'margin': '0 auto'},
                style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial, sans-serif'},
                style_header={'backgroundColor': '#3498DB', 'color': 'white', 'fontWeight': 'bold'}
            ),
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.Button('Add Row', id='add-row-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#1ABC9C',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
            html.Button('Save Table', id='save-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#1ABC9C',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Load Table', style={
                    'width': '300px',
                    'backgroundColor': '#1ABC9C',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '16px',
                    'margin': '10px',
                    'borderRadius': '10px',
                    'cursor': 'pointer'
                }),
                multiple=False,
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'flexWrap': 'wrap'}),

        html.Div([
            html.Button('Calculate GCI', id='calculate-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#3498DB',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
            html.Button('Download GCI Results', id='download-gci-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#3498DB',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
        ], style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'flexWrap': 'wrap'}),

        html.Div([
            dash_table.DataTable(
                id='gci-results-table',
                style_table={'width': '80%', 'margin': '20px auto'},
                style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'Arial, sans-serif'},
                style_header={'backgroundColor': '#3498DB', 'color': 'white', 'fontWeight': 'bold'}
            ),
        ], style={'marginBottom': '20px'}),

        dcc.Download(id="download-table"),
        dcc.Download(id="download-gci")
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})

    return layout
