import dash
from dash import html, dcc, dash_table
import os

def layout_about():
    layout = html.Div([
        html.Iframe(id='pdf-viewer', src=os.path.join('assets', 'about.pdf'),
                    style={'width': '100%', 'height': '600px', 'margin': 'auto',
                           'text-align': 'center', 'display': 'flex', 'justify-content': 'center'})
    ])
    return layout
