import dash
from dash import dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

from dash import html
from dash import dcc
from dash import dash_table


def load_data():
    df = pd.read_excel('setups/Var_Table.xlsx')
    return df.to_dict('records')


def layout_GCI():
    layout = html.Div([
        html.Br(),
        html.H1("Cálculo de GCI (Grid Convergence Index)", style={
            'textAlign': 'center',
            'color': '#2C3E50',
            'fontFamily': 'Arial, sans-serif',
            'fontWeight': 'bold'
        }),
        html.Div([
            html.Label("Índice de Refinamento de Malha:", style={
                'fontSize': '16px',
                'fontWeight': 'bold',
                'marginBottom': '10px'
            }),
            dcc.Input(id='refinement-index', type='number', value=2, step=0.01, style={
                'width': '100px',
                'padding': '10px',
                'fontSize': '16px',
                'borderRadius': '10px',
                'border': '2px solid #3498DB',
                'textAlign': 'center',
                'marginBottom': '20px',
                'marginLeft': '20px'
            })
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            dash_table.DataTable(
                id='editable-table',
                columns=[
                    {'name': 'Variável', 'id': 'variable', 'type': 'text'},
                    {'name': 'Malha Grosseira', 'id': 'coarse', 'type': 'numeric'},
                    {'name': 'Malha Média', 'id': 'medium', 'type': 'numeric'},
                    {'name': 'Malha Fina', 'id': 'fine', 'type': 'numeric'}
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
            html.Button('Adicionar Linha', id='add-row-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#1ABC9C',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
            html.Button('Salvar Tabela', id='save-button', n_clicks=0, style={
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
                children=html.Button('Carregar Tabela', style={
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
            html.Button('Calcular GCI', id='calculate-button', n_clicks=0, style={
                'width': '300px',
                'backgroundColor': '#3498DB',
                'color': 'white',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'margin': '10px',
                'borderRadius': '10px',
                'cursor': 'pointer'
            }),
            html.Button('Baixar Resultados GCI', id='download-gci-button', n_clicks=0, style={
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
