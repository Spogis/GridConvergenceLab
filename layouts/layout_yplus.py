import dash
from dash import html, dcc, dash_table
import os


def layout_yplus():
    layout = html.Div([
        # Título principal
        #html.H2("Yplus Calculator", style={'textAlign': 'center', 'marginBottom': '40px'}),

        # Inputs com título acima
        html.Div([
            html.H3("Input Parameters", style={'textAlign': 'center', 'marginBottom': '30px'}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Label('Density (kg/m³):'),
                        dcc.Input(id='input-density', type='number', min=0, value=1000,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingRight': '10px'}),

                    html.Div([
                        html.Label('Dynamic Viscosity (Pa.s):'),
                        dcc.Input(id='input-viscosity', type='number', min=0, value=0.001,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingRight': '10px'}),

                    html.Div([
                        html.Label('Freestream Velocity (m/s):'),
                        dcc.Input(id='input-velocity', type='number', min=0, value=1,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingRight': '10px'}),
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
                          'width': '100%'}),

                html.Div([
                    html.Div([
                        html.Label('Characteristic Length (m):'),
                        dcc.Input(id='input-length', type='number', min=0, value=1,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingLeft': '10px'}),

                    html.Div([
                        html.Label('Desired Yplus:'),
                        dcc.Input(id='input-yplus', type='number', min=0, value=1,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingLeft': '10px'}),

                    html.Div([
                        html.Label('Growth Rate:'),
                        dcc.Input(id='input-growth-rate', type='number', min=1, value=1.2,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingLeft': '10px'}),
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
                          'width': '100%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'width': '100%'}),
        ], style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '20px', 'borderRadius': '10px',
                  'marginBottom': '40px', 'maxWidth': '800px', 'margin': '0 auto'}),

        html.Br(),

        # Results com título acima
        html.Div([
            html.H3("Results", style={'textAlign': 'center', 'marginBottom': '30px'}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Label('Reynolds:', style={'marginBottom': '10px'}),
                        dcc.Input(id='output-reynolds', type='text', disabled=True,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center',
                                         'backgroundColor': '#ECF0F1'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingRight': '10px'}),

                    html.Div([
                        html.Label('Boundary Layer Thickness:', style={'marginBottom': '10px'}),
                        dcc.Input(id='output-boundary-layer-thickness', type='text', disabled=True,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center',
                                         'backgroundColor': '#ECF0F1'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingRight': '10px'}),
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
                          'width': '100%'}),

                html.Div([
                    html.Div([
                        html.Label('First Layer Thickness:', style={'marginBottom': '10px'}),
                        dcc.Input(id='output-first-layer-thickness', type='text', disabled=True,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center',
                                         'backgroundColor': '#ECF0F1'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingLeft': '10px'}),

                    html.Div([
                        html.Label('Number of Layers:', style={'marginBottom': '10px'}),
                        dcc.Input(id='output-number-of-layers', type='text', disabled=True,
                                  style={'width': '100%', 'padding': '10px', 'textAlign': 'center',
                                         'backgroundColor': '#ECF0F1'}),
                    ], style={'marginBottom': '20px', 'width': '100%', 'paddingLeft': '10px'}),
                ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center',
                          'width': '100%'}),
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'width': '100%'}),
        ], style={'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)', 'padding': '20px', 'borderRadius': '10px',
                  'maxWidth': '800px', 'margin': '0 auto'}),
    ], style={'maxWidth': '900px', 'margin': '0 auto', 'padding': '20px'})

    return layout

