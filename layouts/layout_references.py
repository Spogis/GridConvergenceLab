import dash
from dash import html, dcc, dash_table
import os

def layout_references():
    layout = html.Div(
        style={
            'backgroundColor': '#f0f0f0',
            'padding': '30px',
            'fontFamily': 'Arial, sans-serif',
            'borderRadius': '10px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'
        },
        children=[
            html.Ul(
                style={
                    'listStyleType': 'disc',
                    'paddingLeft': '20px',
                    'margin': '0'
                },
                children=[
                    html.Li(
                        style={'marginBottom': '20px'},
                        children=[
                            html.A(
                                'Celik et al., "Procedure of Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", '
                                'Journal of Fluids Engineering, 130(7), 2008',
                                href='https://doi.org/10.1115/1.2960953',
                                target='_blank',
                                style={
                                    'textDecoration': 'none',
                                    'color': '#007BFF',
                                    'fontSize': '18px'
                                }
                            )
                        ]
                    ),
                    html.Li(
                        style={'marginBottom': '20px'},
                        children=[
                            html.A(
                                'NASA Tutorial on Spatial Convergence in CFD Simulations',
                                href='https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html',
                                target='_blank',
                                style={
                                    'textDecoration': 'none',
                                    'color': '#007BFF',
                                    'fontSize': '18px'
                                }
                            )
                        ]
                    ),
                    html.Li(
                        style={'marginBottom': '20px'},
                        children=[
                            html.A(
                                'Oberkampf and Roy, "Verification and Validation in Scientific Computing", Cambridge University Press, 2013',
                                href='https://doi.org/10.1017/CBO9780511760396',
                                target='_blank',
                                style={
                                    'textDecoration': 'none',
                                    'color': '#007BFF',
                                    'fontSize': '18px'
                                }
                            )
                        ]
                    ),
                    html.Li(
                        style={'marginBottom': '20px'},
                        children=[
                            html.A(
                                'Python Library Used - pyGCS - MIT License',
                                href='https://github.com/tomrobin-teschner/pyGCS',
                                target='_blank',
                                style={
                                    'textDecoration': 'none',
                                    'color': '#007BFF',
                                    'fontSize': '18px'
                                }
                            )
                        ]
                    ),
                ]
            )
        ]
    )

    return layout
