import dash
from dash import dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

def layout_XY_Plot():
    layout = html.Div([
        html.Br(),
        html.H1("Upload de Arquivos Excel e Análise de Dados", style={
            'textAlign': 'center',
            'color': '#2C3E50',
            'fontFamily': 'Arial, sans-serif',
            'fontWeight': 'bold'
        }),
        html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-data-1',
                    children=html.Div([
                        'Arraste e solte ou ',
                        html.A('selecione o arquivo 1', style={'color': '#1ABC9C', 'fontWeight': 'bold'})
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
                html.Div(id='output-file-upload-1', style={'textAlign': 'center', 'marginTop': '10px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Upload(
                    id='upload-data-2',
                    children=html.Div([
                        'Arraste e solte ou ',
                        html.A('selecione o arquivo 2', style={'color': '#1ABC9C', 'fontWeight': 'bold'})
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
                html.Div(id='output-file-upload-2', style={'textAlign': 'center', 'marginTop': '10px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
        html.Br(),
        html.Div([
            html.Button('Gerar Análise', id='analyze-button', n_clicks=0, style={
                'backgroundColor': '#1ABC9C',
                'color': 'white',
                'border': 'none',
                'padding': '15px 30px',
                'textAlign': 'center',
                'textDecoration': 'none',
                'display': 'inline-block',
                'fontSize': '18px',
                'margin': '20px 0',
                'cursor': 'pointer',
                'borderRadius': '10px',
                'fontWeight': 'bold'
            }),
        ], style={'textAlign': 'center'}),
        html.Div(id='output-analysis', style={'margin': '20px', 'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
        dcc.Graph(id='output-graph', style={'margin': '20px'})
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})

    return layout
