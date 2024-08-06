import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

def layout_picture_gray():
    layout = html.Div([
        # html.Br(),
        # html.H1("Image Upload and Similarity Analysis", style={
        #     'textAlign': 'center',
        #     'color': '#2C3E50',
        #     'fontFamily': 'Arial, sans-serif',
        #     'fontWeight': 'bold'
        # }),
        html.Div([
            html.Div([
                dcc.Upload(
                    id='upload-image-1',
                    children=html.Div([
                        'Drag and drop or ',
                        html.A('select image 1', style={'color': '#3498DB', 'fontWeight': 'bold'})
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
                html.Div(id='output-image-upload-1', style={'textAlign': 'center', 'marginTop': '10px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                dcc.Upload(
                    id='upload-image-2',
                    children=html.Div([
                        'Drag and drop or ',
                        html.A('select image 2', style={'color': '#3498DB', 'fontWeight': 'bold'})
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
                html.Div(id='output-image-upload-2', style={'textAlign': 'center', 'marginTop': '10px'})
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'display': 'flex', 'justifyContent': 'center'}),
        html.Br(),
        html.Div([
            html.Button('Generate Analysis', id='analyze-button', n_clicks=0, style={
                'backgroundColor': '#3498DB',
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
        html.Div(id='output-analysis2', style={'margin': '20px', 'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'}),
        dcc.Graph(id='output-graph-1', style={'margin': '20px'}),
        dcc.Graph(id='output-graph-2', style={'margin': '20px'})
    ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})

    return layout
