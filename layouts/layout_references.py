import dash
from dash import html, dcc, dash_table
import os

def layout_references():
    layout = html.Div(children=[
        html.Ul(children=[
            html.Li(children=[
                html.A(
                    'Celik et al., "Procedure of Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications", '
                    'Journal of Fluids Engineering, 130(7), 2008',
                    href='https://doi.org/10.1115/1.2960953',
                    target='_blank'
                )
            ]),
            html.Br(),
            html.Li(children=[
                html.A(
                    'NASA Tutorial on Spatial Convergence in CFD Simulations',
                    href='https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html',
                    target='_blank'
                )
            ]),
            html.Br(),
            html.Li(children=[
                html.A(
                    'Oberkampf and Roy, "Verification and Validation in Scientific Computing", Cambridge University Press, 2013',
                    href='https://doi.org/10.1017/CBO9780511760396',
                    target='_blank'
                )
            ]),
        ]),
    ])

    return layout
