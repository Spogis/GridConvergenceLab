import io
import os
import base64

import pandas as pd
import numpy as np
from math import log10, sqrt

import dash
from dash import Input, Output, State, ctx, dash_table, Patch
from dash import html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim

from PIL import Image

import plotly.graph_objs as go
import plotly.express as px

# Import Layouts
from layouts.layout_about import *
from layouts.layout_GCI import *
from layouts.layout_PlotXY import *
from layouts.layout_Picture_Gray import *
from layouts.layout_Picture_RGB import *
from layouts.layout_GCI_from_curves import *
from layouts.layout_GCI_from_pictures import *
from layouts.layout_references import *
from layouts.layout_GCI_from_curves_averages import *
from layouts.layout_citation import *
from layouts.layout_yplus import *
from layouts.layout_yplus_impeller import *

# Import APPS
from apps.GCI import *
from apps.gci_from_curve_data import *
from apps.stats_from_pics import *
from apps.yplus import *

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "Grid Convergence Lab"

server = app.server

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png',
                 style={'width': '100%', 'height': 'auto', 'margin-left': 'auto',
                        'margin-right': 'auto', 'position': 'fixed', 'top': '0', 'left': '0', 'z-index': '1000'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        html.Div([
            html.Br(),
            dcc.Tabs(id='tabs', value='CGI', children=[
                dcc.Tab(label='Classic GCI', value='CGI',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'margin-top': '20px',
                               'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px', 'margin-top': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='GCI From Curves', value='CGI_from_curves',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='GCI From Curves Statistics', value='CGI_from_curves_averages',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='GCI From Pictures Statistics', value='CGI_from_pictures',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='XY Plot Analysis', value='XY_Plot',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Image Analysis (RGB)', value='Picture_RGB',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Image Analysis (Gray)', value='Picture_Gray',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='y+', value='Yplus',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='y+ for Impellers', value='Yplus_Impeller',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),


                dcc.Tab(label='References', value='References',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='How to cite our work?', value='Citation',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='About', value='About',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'width': '220px',
                      'padding': '10px', 'position': 'fixed',
                      'margin-top': '100px', 'left': '0', 'z-index': '999'}),
        ], style={'display': 'flex'}),
        html.Div(id='tabs-content', style={'flex': 1, 'padding': '10px', 'border-radius': '10px',
                                           'border': '2px solid #ccc',
                                           'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)', 'margin-left': '220px',
                                           'margin-top': '100px', 'overflow-y': 'auto',
                                           'height': 'calc(100vh - 100px)'})
    ], style={'display': 'flex'}),
    dcc.Store(id='store-data'),
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def update_tab_content(selected_tab):
    if selected_tab == 'CGI':
        return layout_GCI()
    if selected_tab == 'CGI_from_curves':
        return layout_GCI_from_curves()
    if selected_tab == 'CGI_from_curves_averages':
        return layout_GCI_from_curves_averages()
    if selected_tab == 'CGI_from_pictures':
        return layout_GCI_from_pictures()
    if selected_tab == 'XY_Plot':
        return layout_XY_Plot()
    if selected_tab == 'Picture_Gray':
        return layout_picture_gray()
    if selected_tab == 'Picture_RGB':
        return layout_picture_rgb()
    if selected_tab == 'Yplus':
        return layout_yplus()
    if selected_tab == 'Yplus_Impeller':
        return layout_yplus_impeller()
    if selected_tab == 'References':
        return layout_references()
    if selected_tab == 'Citation':
        return layout_citation()
    elif selected_tab == 'About':
        return layout_about()

########################################################################################################################
# GCI Callbacks
########################################################################################################################

@app.callback(
    Output('editable-table', 'data'),
    Input('editable-table', 'data')
)
def update_table(data):
    if data is None or len(data) == 0:
        return load_data()
    return data


@app.callback(
    Output('mesh-sizes-table', 'data'),
    Input('mesh-sizes-table', 'data'),
)
def update_mesh_table(data):
    if data is None or len(data) == 0:
        return load_mesh_sizes()
    return data


@app.callback(
    Output('domain-volume', 'value'),
    Input('domain-volume', 'value'),
)
def update_mesh_table(data):
    if data is None:
        return load_volume()
    return data

# Callback to add rows to the table
@app.callback(
    Output('editable-table', 'data', allow_duplicate=True),
    Input('add-row-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('editable-table', 'columns'),
    prevent_initial_call=True
)
def add_row(n_clicks, rows, columns):
    if n_clicks > 0:
        rows.append({c['id']: 0 for c in columns})
    return rows

# Callback to calculate and display GCI results
@app.callback(
    [Output('gci-results-table', 'data'),
     Output('gci-results-table', 'columns')],
    Input('calculate-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('mesh-sizes-table', 'data'),
    State('domain-volume', 'value'),
    State('mesh-type', 'value'),
    prevent_initial_call=True
)
def calculate_gci(n_clicks, rows, mesh, volume, mesh_type):
    if n_clicks > 0:
        variables = [row['variable'] for row in rows]
        results_coarse = [float(row['coarse']) for row in rows]
        results_medium = [float(row['medium']) for row in rows]
        results_fine = [float(row['fine']) for row in rows]

        nodes_coarse = mesh[0]['coarse']
        nodes_medium = mesh[0]['medium']
        nodes_fine = mesh[0]['fine']

        if mesh_type == '3D':
            mesh_type = 3
        if mesh_type == '2D':
            mesh_type = 2

        p_values, gci_medium_values, gci_fine_values, GCI_asymptotic_values,\
            phi_extrapolated_values, r_fine_list, r_medium_list = (
            calculate_gci_multiple_variables(results_coarse, results_medium,
                                             results_fine, nodes_coarse,
                                             nodes_medium, nodes_fine,
                                             volume, mesh_type))

        df_results = pd.DataFrame({
            'Variable': variables,
            'p': p_values,
            'Medium Mesh GCI': gci_medium_values,
            'Fine Mesh GCI': gci_fine_values,
            'GCI Asymptotic': GCI_asymptotic_values,
            'phi extrapolated': phi_extrapolated_values,
            'r fine mesh': r_fine_list,
            'r medium mesh': r_medium_list,
        })

        # Formatando os valores
        df_results['p'] = df_results['p'].map(lambda x: f"{x:.2f}")
        df_results['Medium Mesh GCI'] = df_results['Medium Mesh GCI'].map(lambda x: f"{x:.1%}")
        df_results['Fine Mesh GCI'] = df_results['Fine Mesh GCI'].map(lambda x: f"{x:.1%}")
        df_results['GCI Asymptotic'] = df_results['GCI Asymptotic'].map(lambda x: f"{x:.3f}")
        df_results['phi extrapolated'] = df_results['phi extrapolated'].map(lambda x: f"{x:.3e}")
        df_results['r fine mesh'] = df_results['r fine mesh'].map(lambda x: f"{x:.3f}")
        df_results['r medium mesh'] = df_results['r medium mesh'].map(lambda x: f"{x:.3f}")

        columns = [{'name': col, 'id': col} for col in df_results.columns]
        data = df_results.to_dict('records')

        return data, columns

# Callback to save the variable table
@app.callback(
    Output("editable-table", "data", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    State("editable-table", "data"),
    State("mesh-sizes-table", "data"),
    State('domain-volume', 'value'),
    prevent_initial_call=True,
)
def save_table(n_clicks, rows, mesh, volume):
    df_variables = pd.DataFrame(rows)
    df_mesh_sizes = pd.DataFrame(mesh)
    df_volume = pd.DataFrame({'volume': [volume]})

    if not os.path.exists('setups'):
        os.makedirs('setups')

    # Usando ExcelWriter para salvar os DataFrames em abas diferentes
    file_path = 'setups/Var_Table.xlsx'
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df_variables.to_excel(writer, sheet_name='Variables', index=False)
        df_mesh_sizes.to_excel(writer, sheet_name='MeshSizes', index=False)
        df_volume.to_excel(writer, sheet_name='Volume', index=False)

    return rows

# Callback to load the variable table
@app.callback(
    Output('editable-table', 'data', allow_duplicate=True),
    Output('mesh-sizes-table', 'data', allow_duplicate=True),
    Output('domain-volume', 'value', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def load_table(contents, filename):
    if contents is None:
        return []
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded), sheet_name='Variables')
    df2 = pd.read_excel(io.BytesIO(decoded), sheet_name='MeshSizes')
    df3 = pd.read_excel(io.BytesIO(decoded), sheet_name='Volume')
    return df.to_dict('records'), df2.to_dict('records'), df3['volume'][0]

# Callback to download GCI results
@app.callback(
    Output("download-gci", "data"),
    Input("download-gci-button", "n_clicks"),
    State("gci-results-table", "data"),
    prevent_initial_call=True
)
def download_gci_results(n_clicks, rows):
    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='GCI_Results')
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), "GCI_Results.xlsx")

########################################################################################################################
# XY Plot Callbacks
########################################################################################################################

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_excel(io.BytesIO(decoded))


@app.callback([Output('output-file-upload-1', 'children')],
              [Input('upload-data-1', 'filename')])
def update_filenames1(filename1):
    return [f'File 1: {filename1}']


@app.callback([Output('output-file-upload-2', 'children')],
              [Input('upload-data-2', 'filename')])
def update_filename(filename2):
    return [f'File 1: {filename2}']


@app.callback([Output('output-file-upload-3', 'children')],
              [Input('upload-data-3', 'filename')])
def update_filenames3(filename3):
    return [f'File 3: {filename3}']


@app.callback(
    [Output('editable-table', 'data', allow_duplicate=True),
     Output('xy-data-graph', 'figure', allow_duplicate=True),
     Output('xy-data-graph', 'style', allow_duplicate=True)],
    [Input('import-data-button', 'n_clicks')],
    [State('upload-data-1', 'contents'),
     State('upload-data-2', 'contents'),
     State('upload-data-3', 'contents'),
     State('splits', 'value'),
     ],
    prevent_initial_call=True)


def import_data_from_curves(n_clicks, contents1, contents2, contents3, splits):
    if n_clicks > 0 and contents1 and contents2 and contents3:
        content1_type, content1_string = contents1.split(',')
        decoded1 = base64.b64decode(content1_string)

        content2_type, content2_string = contents2.split(',')
        decoded2 = base64.b64decode(content2_string)

        content3_type, content3_string = contents3.split(',')
        decoded3 = base64.b64decode(content3_string)

        df_coarse = pd.read_excel(io.BytesIO(decoded1))
        df_medium = pd.read_excel(io.BytesIO(decoded2))
        df_fine = pd.read_excel(io.BytesIO(decoded3))

        gci_from_curve_data(df_coarse, df_medium, df_fine, splits)
        df_data = pd.read_excel('data/curve_for_gci.xlsx')


        ### Plot data from curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['coarse'], mode='lines', name='Coarse Mesh'))
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['medium'], mode='lines', name='Medium Mesh'))
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['fine'], mode='lines', name='Fine Mesh'))

        # Obtendo os valores mínimo e máximo de x para definir os limites
        x_min = df_data['x'].min()
        x_max = df_data['x'].max()

        # Definindo explicitamente os valores dos ticks
        tick_vals = df_data['x']

        fig.update_layout(
            title='XY Imported Data',
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(
                range=[x_min, x_max],  # Define os limites do eixo x.
                tickvals=tick_vals,  # Define explicitamente os valores dos ticks.
                tickformat='.1f',  # Define o formato dos ticks. '.1f' formata com uma casa decimal.
                nticks=10  # Define o número de ticks. Ajuste conforme necessário.
            ),
            width=1000,
        )

        # Excluindo a coluna 'x'
        df_data = df_data.drop('x', axis=1)

        return [df_data.to_dict('records'), fig, {'display': 'block'}]
    return [[], go.Figure(), {'display': 'none'}]

@app.callback(
    [Output('output-analysis', 'children'),
     Output('output-graph', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [State('upload-data-1', 'contents'),
     State('upload-data-1', 'filename'),
     State('upload-data-2', 'contents'),
     State('upload-data-2', 'filename')]
)
def update_output(n_clicks, contents1, filename1, contents2, filename2):
    if n_clicks > 0 and contents1 and contents2:
        df1 = parse_contents(contents1)
        df2 = parse_contents(contents2)

        # Interpolação
        x_common = np.linspace(min(df1['x'].min(), df2['x'].min()), max(df1['x'].max(), df2['x'].max()), 100)
        spline1 = interp1d(df1['x'], df1['y'], kind='cubic', fill_value="extrapolate")
        spline2 = interp1d(df2['x'], df2['y'], kind='cubic', fill_value="extrapolate")
        y1_interp = spline1(x_common)
        y2_interp = spline2(x_common)
        df1_interp = pd.DataFrame({'x': x_common, 'y': y1_interp})
        df2_interp = pd.DataFrame({'x': x_common, 'y': y2_interp})

        # Métricas de Similaridade
        similarity_cosine = cosine_similarity([df1_interp['y']], [df2_interp['y']])[0, 0]
        distance_euclidean = euclidean(df1_interp['y'], df2_interp['y'])
        correlation_pearson, _ = pearsonr(df1_interp['y'], df2_interp['y'])
        correlation_spearman, _ = spearmanr(df1_interp['y'], df2_interp['y'])

        # Cálculo de outras métricas de ajuste
        residuals = df1_interp['y'] - df2_interp['y']
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)

        # Cálculo do R²
        total_sum_of_squares = np.sum((df1_interp['y'] - np.mean(df1_interp['y']))**2)
        r_squared = 1 - (np.sum(residuals ** 2) / total_sum_of_squares)

        # Análise Interpretativa usando R²
        analysis_interpretation = ""
        if r_squared > 0.9:
            analysis_interpretation = "The curves have a very similar shape."
        elif r_squared > 0.75:
            analysis_interpretation = "The curves have a quite similar shape."
        elif r_squared > 0.5:
            analysis_interpretation = "The curves have a moderately similar shape."
        elif r_squared > 0:
            analysis_interpretation = "The curves have a slightly similar shape."
        else:
            analysis_interpretation = "The curves are not similar in shape."


        # Resultados da Análise
        result_text = [
            html.H4(f'Cosine Similarity: {similarity_cosine:.4f}'),
            html.H4(f'Euclidean Distance: {distance_euclidean:.4f}'),
            html.H4(f'Pearson Correlation: {correlation_pearson:.4f}'),
            html.H4(f'Spearman Correlation: {correlation_spearman:.4f}'),
            html.H4(f'R-squared: {r_squared:.4f}'),
            html.H4(f'MAE: {mae:.4f}'),
            html.H4(f'MSE: {mse:.4f}'),
            html.H4(f'RMSE: {rmse:.4f}'),
            #html.H4(f'Analysis: {analysis_interpretation}')
        ]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1['x'], y=df1['y'], mode='markers', name='Original Data File 1'))
        fig.add_trace(go.Scatter(x=df2['x'], y=df2['y'], mode='markers', name='Original Data File 2'))
        fig.add_trace(go.Scatter(x=df1_interp['x'], y=df1_interp['y'], mode='lines', name='Interpolated File 1'))
        fig.add_trace(go.Scatter(x=df2_interp['x'], y=df2_interp['y'], mode='lines', name='Interpolated File 2'))
        fig.update_layout(title='Interpolated Data Comparison',
                          xaxis_title='X',
                          yaxis_title='Y')

        return result_text, fig
    return html.Div("Upload files and click the button to analyze"), go.Figure()



########################################################################################################################
# Callbacks Picture Gray
########################################################################################################################

@app.callback(
    [Output('output-image-upload-1', 'children'),
     Output('output-image-upload-2', 'children')],
    [Input('upload-image-1', 'filename'),
     Input('upload-image-2', 'filename')]
)
def update_filenames(filename1, filename2):
    return f'Image 1: {filename1}', f'Image 2: {filename2}'

def parse_image(contents):
    return Image.open(io.BytesIO(base64.b64decode(contents.split(',')[1])))

def psnr(img1, img2):
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:  # Significa que não há diferença entre as imagens
        return 100
    max_pixel = 255.0
    psnr_value = 20 * log10(max_pixel / sqrt(mse))
    return psnr_value

def ncc(img1, img2):
    img1_mean_subtracted = img1 - np.mean(img1)
    img2_mean_subtracted = img2 - np.mean(img2)
    numerator = np.sum(img1_mean_subtracted * img2_mean_subtracted)
    denominator = np.sqrt(np.sum(img1_mean_subtracted ** 2) * np.sum(img2_mean_subtracted ** 2))
    return numerator / denominator

@app.callback(
    [Output('output-analysis2', 'children'),
     Output('output-graph-1', 'figure'),
     Output('output-graph-2', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [State('upload-image-1', 'contents'),
     State('upload-image-1', 'filename'),
     State('upload-image-2', 'contents'),
     State('upload-image-2', 'filename')]
)
def update_output(n_clicks, contents1, filename1, contents2, filename2):
    if n_clicks > 0 and contents1 and contents2:
        img1 = parse_image(contents1)
        img2 = parse_image(contents2)

        # Convert to grayscale
        gray1 = img1.convert('L')
        gray2 = img2.convert('L')

        # Resize the second image to match the first image's size
        gray2_resized = gray2.resize(gray1.size)

        # Convert images to numpy arrays
        gray1_array = np.array(gray1)
        gray2_resized_array = np.array(gray2_resized)

        # Calculate MSE
        mse_value = mean_squared_error(gray1_array.flatten(), gray2_resized_array.flatten())

        # Calculate SSIM
        ssim_value, _ = ssim(gray1_array, gray2_resized_array, full=True)

        # Calculate Pearson Correlation
        pearson_corr, _ = pearsonr(gray1_array.flatten(), gray2_resized_array.flatten())

        # Calculate PSNR
        psnr_value = psnr(gray1_array, gray2_resized_array)

        # Calculate NCC
        ncc_value = ncc(gray1_array, gray2_resized_array)

        # Interpretative analysis based on SSIM, PSNR, and NCC
        analysis_interpretation = ""
        if ssim_value > 0.9 and psnr_value > 30 and ncc_value > 0.9:
            analysis_interpretation = "The images are very similar."
        elif ssim_value > 0.75 and psnr_value > 25 and ncc_value > 0.75:
            analysis_interpretation = "The images are quite similar."
        elif ssim_value > 0.5 and psnr_value > 20 and ncc_value > 0.5:
            analysis_interpretation = "The images are moderately similar."
        else:
            analysis_interpretation = "The images have no similarity."

        spogis_value = mse_value / (ssim_value * psnr_value * ncc_value)
        # Analysis results
        result_text = [
            html.H4(f'Mean Squared Error: {mse_value:.4f}'),
            html.H4(f'Structural Similarity Index (SSIM): {ssim_value:.4f}'),
            html.H4(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f}'),
            html.H4(f'Normalized Cross-Correlation (NCC): {ncc_value:.4f}'),
            html.H4(f'Spogis Number: {spogis_value:.4f}'),
            #html.H4(f'Analysis: {analysis_interpretation}')
        ]

        # Display the images using Plotly Express in grayscale
        fig1 = px.imshow(gray1_array, color_continuous_scale='gray', title="Image 1")
        fig2 = px.imshow(gray2_resized_array, color_continuous_scale='gray', title="Image 2")

        return result_text, fig1, fig2

    return html.Div("Upload images and click the button to analyze"), {}, {}

########################################################################################################################
#Callbacks Picture RGB
########################################################################################################################

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

@app.callback(
    [Output('output-image-upload-1-rgb', 'children'),
     Output('output-image-upload-2-rgb', 'children')],
    [Input('upload-image-1-rgb', 'filename'),
     Input('upload-image-2-rgb', 'filename')]
)
def update_filenames(filename1, filename2):
    return f'Image 1: {filename1}', f'Image 2: {filename2}'

@app.callback(
    [Output('output-analysis-rgb', 'children'),
     Output('output-graph-1-rgb', 'figure'),
     Output('output-graph-2-rgb', 'figure'),
     Output('output-diff-rgb', 'figure')],
    [Input('analyze-button', 'n_clicks')],
    [State('upload-image-1-rgb', 'contents'),
     State('upload-image-1-rgb', 'filename'),
     State('upload-image-2-rgb', 'contents'),
     State('upload-image-2-rgb', 'filename')]
)
def update_output(n_clicks, contents1, filename1, contents2, filename2):
    if n_clicks > 0 and contents1 and contents2:
        img1 = parse_image(contents1)
        img2 = parse_image(contents2)

        # Redimensionar a segunda imagem
        img2_resized = img2.resize(img1.size)

        # Calcular MSE
        def mse(imageA, imageB):
            arrA = np.array(imageA)
            arrB = np.array(imageB)
            err = np.sum((arrA - arrB) ** 2)
            err /= float(arrA.shape[0] * arrA.shape[1] * arrA.shape[2])
            return err

        mse_value = mse(img1, img2_resized)

        # Calcular SSIM
        img1_array = np.array(img1)
        img2_resized_array = np.array(img2_resized)

        ssim_value, _ = ssim(img1_array, img2_resized_array, multichannel=True, channel_axis=-1, full=True)

        # Calcular Correlação de Pearson
        pearson_corr, _ = pearsonr(img1_array.flatten(), img2_resized_array.flatten())

        # Calcular PSNR
        def psnr(imageA, imageB):
            mse_value = mse(imageA, imageB)
            if mse_value == 0:
                return 100
            PIXEL_MAX = 255.0
            return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_value))

        psnr_value = psnr(img1, img2_resized)

        # Calcular NCC
        def ncc(imageA, imageB):
            arrA = (np.array(imageA) - np.mean(imageA)).flatten()
            arrB = (np.array(imageB) - np.mean(imageB)).flatten()
            return np.sum(arrA * arrB) / np.sqrt(np.sum(arrA ** 2) * np.sum(arrB ** 2))

        ncc_value = ncc(img1, img2_resized)

        # Interpretative analysis based on SSIM, PSNR, and NCC
        analysis_interpretation = ""
        if ssim_value > 0.9 and psnr_value > 30 and ncc_value > 0.9:
            analysis_interpretation = "The images are very similar."
        elif ssim_value > 0.75 and psnr_value > 25 and ncc_value > 0.75:
            analysis_interpretation = "The images are quite similar."
        elif ssim_value > 0.5 and psnr_value > 20 and ncc_value > 0.5:
            analysis_interpretation = "The images are moderately similar."
        else:
            analysis_interpretation = "The images have no similarity."

        spogis_value = mse_value/(ssim_value*psnr_value*ncc_value)
        # Resultados da análise
        result_text = [
            html.H4(f'Mean Squared Error: {mse_value:.4f}'),
            html.H4(f'Structural Similarity Index (SSIM): {ssim_value:.4f}'),
            html.H4(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.4f}'),
            html.H4(f'Normalized Cross-Correlation (NCC): {ncc_value:.4f}'),
            html.H4(f'Spogis Number: {spogis_value:.4f}'),
            #html.H4(f'Analysis: {analysis_interpretation}')
        ]

        # Exibir as imagens usando Plotly Express
        fig1 = px.imshow(img1_array, title="Image 1")
        fig2 = px.imshow(img2_resized_array, title="Image 2")

        # Calcular a diferença local entre as imagens usando a distância Euclidiana
        #diff_array = np.linalg.norm(img1_array - img2_resized_array, axis=-1)

        # Converter as imagens para arrays numpy
        img1_array = np.array(img1)
        img2_resized_array = np.array(img2_resized)

        # Converter para o espaço de cinza
        img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_resized_array, cv2.COLOR_RGB2GRAY)

        # Calcular o SSIM entre as duas imagens
        ssim_index, diff = ssim(img1_gray, img2_gray, full=True)
        diff_array = (diff * 255).astype(np.uint8)
        diff_array = 255 - diff_array

        # Definir um delta para ignorar pequenas diferenças
        delta = 50  # Ajuste este valor conforme necessário

        # Aplicar o delta (margem de segurança)
        diff_array[diff_array < delta] = 0

        # Exibir as imagens usando Plotly Express
        fig1 = px.imshow(img1_array, title="Image 1")
        fig2 = px.imshow(img2_resized_array, title="Image 2")
        fig_diff = px.imshow(diff_array, title="Difference between Images", color_continuous_scale='viridis')

        return result_text, fig1, fig2, fig_diff

    return html.Div("Upload images and click the button to analyze"), {}, {}, {}

########################################################################################################################
#Callbacks Picture CGI
########################################################################################################################

@app.callback(
    Output('editable-table', 'data', allow_duplicate=True),
    Input('import-pictures-data-button', 'n_clicks'),
    State('upload-data-1', 'contents'),
    State('upload-data-2', 'contents'),
    State('upload-data-3', 'contents'),
    prevent_initial_call=True
)
def import_data_from_curves(n_clicks, contents1, contents2, contents3):
    if n_clicks > 0 and contents1 and contents2 and contents3:
        def base64_to_cv2_image(contents):
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            np_arr = np.frombuffer(decoded, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Falha ao decodificar a imagem")
            return img

        img_coarse = base64_to_cv2_image(contents1)
        img_medium = base64_to_cv2_image(contents2)
        img_fine = base64_to_cv2_image(contents3)

        df_coarse = gci_from_picture_data(img_coarse)
        df_medium = gci_from_picture_data(img_medium)
        df_fine = gci_from_picture_data(img_fine)

        df_data = df_coarse.copy()
        df_data.rename(columns={'value': 'coarse'}, inplace=True)

        df_data['medium'] = df_medium['value']
        df_data['fine'] = df_fine['value']

        df_data.to_excel('setups/Var_Table_GCI_Pictures.xlsx')
        df = pd.read_excel('setups/Var_Table_GCI_Pictures.xlsx', index_col=None)
        return df.to_dict('records')
    return []


########################################################################################################################
#CGI from Curves Averages
########################################################################################################################

@app.callback(
    [Output('editable-table', 'data', allow_duplicate=True),
     Output('xy-data-graph', 'figure', allow_duplicate=True),
     Output('xy-data-graph', 'style', allow_duplicate=True)],
    [Input('import-data-button-averages', 'n_clicks')],
    [State('upload-data-1', 'contents'),
     State('upload-data-2', 'contents'),
     State('upload-data-3', 'contents'),
     State('splits', 'value'),
     ],
    prevent_initial_call=True)


def import_data_from_curves(n_clicks, contents1, contents2, contents3, splits):
    if n_clicks > 0 and contents1 and contents2 and contents3:
        content1_type, content1_string = contents1.split(',')
        decoded1 = base64.b64decode(content1_string)

        content2_type, content2_string = contents2.split(',')
        decoded2 = base64.b64decode(content2_string)

        content3_type, content3_string = contents3.split(',')
        decoded3 = base64.b64decode(content3_string)

        df_coarse = pd.read_excel(io.BytesIO(decoded1))
        df_medium = pd.read_excel(io.BytesIO(decoded2))
        df_fine = pd.read_excel(io.BytesIO(decoded3))

        gci_from_curve_data(df_coarse, df_medium, df_fine, splits)
        df_data = pd.read_excel('data/curve_for_gci.xlsx')


        ### Plot data from curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['coarse'], mode='lines', name='Coarse Mesh'))
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['medium'], mode='lines', name='Medium Mesh'))
        fig.add_trace(go.Scatter(x=df_data['x'], y=df_data['fine'], mode='lines', name='Fine Mesh'))

        # Obtendo os valores mínimo e máximo de x para definir os limites
        x_min = df_data['x'].min()
        x_max = df_data['x'].max()

        # Definindo explicitamente os valores dos ticks
        tick_vals = df_data['x']

        fig.update_layout(
            title='XY Imported Data',
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(
                range=[x_min, x_max],  # Define os limites do eixo x.
                tickvals=tick_vals,  # Define explicitamente os valores dos ticks.
                tickformat='.1f',  # Define o formato dos ticks. '.1f' formata com uma casa decimal.
                nticks=10  # Define o número de ticks. Ajuste conforme necessário.
            ),
            width=1000,
        )

        # Excluindo a coluna 'x'
        df_data = df_data.drop('x', axis=1)

        mean_coarse = df_data['coarse'].mean()
        std_coarse = df_data['coarse'].std()
        var_coarse = df_data['coarse'].var()
        cv_coarse = std_coarse/mean_coarse

        mean_medium = df_data['medium'].mean()
        std_medium = df_data['medium'].std()
        var_medium = df_data['medium'].var()
        cv_medium = std_medium / mean_medium

        mean_fine = df_data['fine'].mean()
        std_fine = df_data['fine'].std()
        var_fine = df_data['fine'].var()
        cv_fine = std_fine / mean_fine


        # Criar o DataFrame com as estatísticas
        data = {
            'variable': ['Mean', 'Standard Deviation', 'Variance', 'Coefficient of Variation'],
            'coarse': [mean_coarse, std_coarse, var_coarse, cv_coarse],
            'medium': [mean_medium, std_medium, var_medium, cv_medium],
            'fine': [mean_fine, std_fine, var_fine, cv_fine]
        }

        df_stats = pd.DataFrame(data)

        return [df_stats.to_dict('records'), fig, {'display': 'block'}]
    return [[], go.Figure(), {'display': 'none'}]


########################################################################################################################
#YPlus
########################################################################################################################

@app.callback([Output('output-reynolds', 'value'),
               Output('output-first-layer-thickness', 'value'),
               Output('output-boundary-layer-thickness', 'value'),
               Output('output-number-of-layers', 'value')],
              [Input('input-density', 'value'),
               Input('input-viscosity', 'value'),
               Input('input-velocity', 'value'),
               Input('input-length', 'value'),
               Input('input-yplus', 'value'),
               Input('input-growth-rate', 'value')])
def calculate_yplus(density, viscosity, freestream_velocity, characteristic_length, desired_yplus, growth_rate):
    Reynolds, DeltaY, Boundary_Layer_Thickness, Number_Of_Layers = yplus(density=density, viscosity=viscosity, freestream_velocity=freestream_velocity,
                   desired_yplus=desired_yplus, growth_rate=growth_rate,
                   characteristic_length=characteristic_length, option='Free')

    Reynolds = "{:1.2e}".format(Reynolds)
    DeltaY = "{:.3e}".format(DeltaY)
    Boundary_Layer_Thickness = "{:.3e}".format(Boundary_Layer_Thickness)

    return Reynolds, DeltaY, Boundary_Layer_Thickness, Number_Of_Layers

@app.callback([Output('output-reynolds-impeller', 'value'),
               Output('output-first-layer-thickness-impeller', 'value'),
               Output('output-boundary-layer-thickness-impeller', 'value'),
               Output('output-number-of-layers-impeller', 'value')],
              [Input('input-density-impeller', 'value'),
               Input('input-viscosity-impeller', 'value'),
               Input('input-rpm-impeller', 'value'),
               Input('input-diameter-impeller', 'value'),
               Input('input-yplus-impeller', 'value'),
               Input('input-growth-rate-impeller', 'value')])
def calculate_yplus(density, viscosity, rpm, diameter, desired_yplus, growth_rate):
    Reynolds, DeltaY, Boundary_Layer_Thickness, Number_Of_Layers = yplus(density=density, viscosity=viscosity, rpm=rpm,
                   desired_yplus=desired_yplus, growth_rate=growth_rate,
                   diameter=diameter, option='Impeller')

    Reynolds = "{:1.2e}".format(Reynolds)
    DeltaY = "{:.3e}".format(DeltaY)
    Boundary_Layer_Thickness = "{:.3e}".format(Boundary_Layer_Thickness)

    return Reynolds, DeltaY, Boundary_Layer_Thickness, Number_Of_Layers

if __name__ == '__main__':
    app.run_server(debug=False)
