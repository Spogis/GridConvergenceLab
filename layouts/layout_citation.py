import dash
from dash import html, dcc, dash_table
import os

def layout_citation():
    layout = html.Div(
        children=[
            html.Br(),
            html.Div(
                style={
                    'backgroundColor': '#f9f9f9',
                    'padding': '20px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
                    'maxWidth': '800px',
                    'margin': 'auto',
                    'fontFamily': 'Arial, sans-serif'
                },
                children=[
                    dcc.Markdown(
                        '''
                        ```bibtex
                        @misc{gridconvergencelab,
                            author = {SPOGIS, N., FONTOURA, D. V. R.},
                            title = {Grid Convergence Lab Toolkit},
                            subtitle = {A Python package for Grid Convergence Index Analysis},
                            note = "https://github.com/Spogis/GridConvergeLab",
                            year = {2024},
                        }
                        ```
                        ''',
                        style={
                            'backgroundColor': '#fff',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'boxShadow': '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
                            'fontSize': '16px',
                            'lineHeight': '1.5',
                            'whiteSpace': 'pre-wrap',
                            'border': '1px solid #ddd',
                            'fontFamily': 'Courier New, monospace'
                        }
                    ),
                    html.Br(),
                    dcc.Markdown(
                        '''
                        ```bibtex
                        @software{nicolas_spogis_2024_13288605,
                            author       = {Nicolas Spogis and
                                            Diener, Volpin Ribeiro Fontoura},
                            title        = {Spogis/GridConvergenceLab: v.1.0.1},
                            month        = aug,
                            year         = 2024,
                            publisher    = {Zenodo},
                            version      = {v.1.0.1},
                            doi          = {10.5281/zenodo.13288605},
                            url          = {https://doi.org/10.5281/zenodo.13288605}
                        }
                        ```
                        ''',
                        style={
                            'backgroundColor': '#fff',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'boxShadow': '0 2px 4px 0 rgba(0, 0, 0, 0.1)',
                            'fontSize': '16px',
                            'lineHeight': '1.5',
                            'whiteSpace': 'pre-wrap',
                            'border': '1px solid #ddd',
                            'fontFamily': 'Courier New, monospace'
                        }
                    ),
                    html.P(
                        "Copy the above BibTeX entry for your work.",
                        style={
                            'textAlign': 'center',
                            'fontSize': '18px',
                            'marginTop': '20px',
                            'color': '#555'
                        }
                    )
                ]
            )
        ]
    )

    return layout
