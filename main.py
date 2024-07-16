import dash
from dash import Input, Output, State, ctx, dash_table, Patch
from dash import html
from dash import dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import plotly.graph_objs as go
import plotly.express as px

from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr


import io
import os
import base64


#Import Layouts
from layouts.layout_about import *
from layouts.layout_GCI import *
from layouts.layout_PlotXY import *
from layouts.layout_Picture_Gray import *
from layouts.layout_Picture_RGB import *

#Import APPS
from apps.GCI import *

# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = " Grid Convergence Index (GCI)"

server = app.server


app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='CGI', children=[
            dcc.Tab(label='GCI', value='CGI',
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#ECF0F1'},
                    selected_style={'fontSize': '16px', 'fontWeight': 'bold', 'backgroundColor': '#3498DB', 'color': 'white', 'padding': '10px'}),

            dcc.Tab(label='Gráfico XY', value='XY_Plot',
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#ECF0F1'},
                    selected_style={'fontSize': '16px', 'fontWeight': 'bold', 'backgroundColor': '#3498DB', 'color': 'white', 'padding': '10px'}),

            dcc.Tab(label='Análise de Figuras (Gray)', value='Picture_Gray',
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#ECF0F1'},
                    selected_style={'fontSize': '16px', 'fontWeight': 'bold', 'backgroundColor': '#3498DB', 'color': 'white', 'padding': '10px'}),

            dcc.Tab(label='Análise de Figuras (RGB)', value='Picture_RGB',
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#ECF0F1'},
                    selected_style={'fontSize': '16px', 'fontWeight': 'bold', 'backgroundColor': '#3498DB', 'color': 'white', 'padding': '10px'}),

            dcc.Tab(label='Sobre', value='About',
                    style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '10px', 'backgroundColor': '#ECF0F1'},
                    selected_style={'fontSize': '16px', 'fontWeight': 'bold', 'backgroundColor': '#3498DB', 'color': 'white', 'padding': '10px'}),
        ], style={'width': '80%', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'}),
    ], style={'text-align': 'center'}),
    dcc.Store(id='store-data'),
    html.Div(id='tabs-content', style={'width': '80%', 'margin': '20px auto', 'fontFamily': 'Arial, sans-serif'}),
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#F8F9F9', 'padding': '20px'})




@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def update_tab_content(selected_tab):
    if selected_tab == 'CGI':
        return layout_GCI()
    if selected_tab == 'XY_Plot':
        return layout_XY_Plot()
    if selected_tab == 'Picture_Gray':
        return layout_picture_gray()
    if selected_tab == 'Picture_RGB':
        return layout_picture_rgb()
    elif selected_tab == 'About':
        return layout_about()



########################################################################################################################
#Callbacks GCI
########################################################################################################################

@app.callback(
    Output('editable-table', 'data'),
    Input('editable-table', 'data')
)
def update_table(data):
    if data is None or len(data) == 0:
        return load_data()
    return data

# Callback para adicionar linhas na tabela
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


# Callback para calcular e exibir os resultados do GCI
@app.callback(
    [Output('gci-results-table', 'data'),
     Output('gci-results-table', 'columns')],
    Input('calculate-button', 'n_clicks'),
    State('editable-table', 'data'),
    State('refinement-index', 'value'),
    prevent_initial_call=True
)
def calculate_gci(n_clicks, rows, refine_factor):
    print(refine_factor)
    if n_clicks > 0:
        variables = [row['variable'] for row in rows]
        results_coarse = [float(row['coarse']) for row in rows]
        results_medium = [float(row['medium']) for row in rows]
        results_fine = [float(row['fine']) for row in rows]

        r = refine_factor  # Fator de refinamento

        p_values, gci_medium_values, gci_fine_values = calculate_gci_multiple_variables(results_coarse, results_medium,
                                                                                        results_fine, r)

        df_results = pd.DataFrame({
            'Variável': variables,
            'p': p_values,
            'GCI Malha Média': gci_medium_values,
            'GCI Malha Fina': gci_fine_values,
        })

        columns = [{'name': col, 'id': col} for col in df_results.columns]
        data = df_results.to_dict('records')

        return data, columns


# Callback para salvar a tabela de variáveis
@app.callback(
    Output("editable-table", "data", allow_duplicate=True),
    Input("save-button", "n_clicks"),
    State("editable-table", "data"),
    prevent_initial_call=True,
)
def save_table(n_clicks, rows):
    df = pd.DataFrame(rows)
    if not os.path.exists('setups'):
        os.makedirs('setups')
    df.to_excel('setups/Var_Table.xlsx', index=False)
    return rows


# Callback para carregar a tabela de variáveis
@app.callback(
    Output('editable-table', 'data', allow_duplicate=True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def load_table(contents, filename):
    if contents is None:
        return []
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded))
    return df.to_dict('records')


# Callback para baixar os resultados do GCI
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
#Callbacks PLOT XY
########################################################################################################################

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_excel(io.BytesIO(decoded))

@app.callback(
    [Output('output-file-upload-1', 'children'),
     Output('output-file-upload-2', 'children')],
    [Input('upload-data-1', 'filename'),
     Input('upload-data-2', 'filename')]
)
def update_filenames(filename1, filename2):
    return f'Arquivo 1: {filename1}', f'Arquivo 2: {filename2}'

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

        # Análise interpretativa
        analysis_interpretation = ""
        if similarity_cosine > 0.9 and correlation_pearson > 0.9:
            analysis_interpretation = "As curvas são muito semelhantes."
        elif similarity_cosine > 0.75 and correlation_pearson > 0.75:
            analysis_interpretation = "As curvas são bastante semelhantes."
        elif similarity_cosine > 0.5 and correlation_pearson > 0.5:
            analysis_interpretation = "As curvas são moderadamente semelhantes."
        else:
            analysis_interpretation = "As curvas são pouco semelhantes."

        # Resultados da análise
        result_text = [
            html.H4(f'Similaridade cosseno: {similarity_cosine:.4f}'),
            html.H4(f'Distância Euclidiana: {distance_euclidean:.4f}'),
            html.H4(f'Correlação de Pearson: {correlation_pearson:.4f}'),
            html.H4(f'Análise: {analysis_interpretation}')
        ]

        # Gráfico
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df1['x'], y=df1['y'], mode='markers', name='Dados Originais Arquivo 1'))
        fig.add_trace(go.Scatter(x=df2['x'], y=df2['y'], mode='markers', name='Dados Originais Arquivo 2'))
        fig.add_trace(go.Scatter(x=df1_interp['x'], y=df1_interp['y'], mode='lines', name='Interpolado Arquivo 1'))
        fig.add_trace(go.Scatter(x=df2_interp['x'], y=df2_interp['y'], mode='lines', name='Interpolado Arquivo 2'))
        fig.update_layout(title='Comparação de Dados Interpolados',
                          xaxis_title='X',
                          yaxis_title='Y')

        return result_text, fig
    return html.Div("Carregue os arquivos e clique no botão para analisar"), go.Figure()


########################################################################################################################
#Callbacks Picture Gray
########################################################################################################################

def parse_image(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return Image.open(io.BytesIO(decoded))

@app.callback(
    [Output('output-image-upload-1', 'children'),
     Output('output-image-upload-2', 'children')],
    [Input('upload-image-1', 'filename'),
     Input('upload-image-2', 'filename')]
)
def update_filenames(filename1, filename2):
    return f'Imagem 1: {filename1}', f'Imagem 2: {filename2}'

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

        # Converter para escala de cinza
        gray1 = img1.convert('L')
        gray2 = img2.convert('L')

        # Redimensionar a segunda imagem
        gray2_resized = gray2.resize(gray1.size)

        # Calcular MSE
        def mse(imageA, imageB):
            # Converter as imagens para arrays numpy
            arrA = np.array(imageA)
            arrB = np.array(imageB)
            # Calcular o erro quadrático médio
            err = np.sum((arrA - arrB) ** 2)
            err /= float(arrA.shape[0] * arrA.shape[1])
            return err

        mse_value = mse(gray1, gray2_resized)

        # Calcular SSIM
        gray1_array = np.array(gray1)
        gray2_resized_array = np.array(gray2_resized)

        ssim_value, _ = ssim(gray1_array, gray2_resized_array, full=True)

        # Calcular Correlação de Pearson
        pearson_corr, _ = pearsonr(gray1_array.flatten(), gray2_resized_array.flatten())

        # Análise interpretativa
        analysis_interpretation = ""
        if ssim_value > 0.9 and pearson_corr > 0.9:
            analysis_interpretation = "As imagens são muito semelhantes."
        elif ssim_value > 0.75 and pearson_corr > 0.75:
            analysis_interpretation = "As imagens são bastante semelhantes."
        elif ssim_value > 0.5 and pearson_corr > 0.5:
            analysis_interpretation = "As imagens são moderadamente semelhantes."
        else:
            analysis_interpretation = "As imagens são pouco semelhantes."

        # Resultados da análise
        result_text = [
            html.H4(f'Mean Squared Error: {mse_value:.4f}'),
            html.H4(f'Structural Similarity Index: {ssim_value:.4f}'),
            html.H4(f'Correlação de Pearson: {pearson_corr:.4f}'),
            html.H4(f'Análise: {analysis_interpretation}')
        ]

        # Exibir as imagens usando Plotly Express em escala de cinza
        fig1 = px.imshow(gray1_array, color_continuous_scale='gray', title="Imagem 1")
        fig2 = px.imshow(gray2_resized_array, color_continuous_scale='gray', title="Imagem 2")

        return result_text, fig1, fig2

    return html.Div("Carregue as imagens e clique no botão para analisar"), {}, {}


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
    return f'Imagem 1: {filename1}', f'Imagem 2: {filename2}'

@app.callback(
    [Output('output-analysis-rgb', 'children'),
     Output('output-graph-1-rgb', 'figure'),
     Output('output-graph-2-rgb', 'figure')],
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

        # Análise interpretativa
        analysis_interpretation = ""
        if ssim_value > 0.9 and pearson_corr > 0.9:
            analysis_interpretation = "As imagens são muito semelhantes."
        elif ssim_value > 0.75 and pearson_corr > 0.75:
            analysis_interpretation = "As imagens são bastante semelhantes."
        elif ssim_value > 0.5 and pearson_corr > 0.5:
            analysis_interpretation = "As imagens são moderadamente semelhantes."
        else:
            analysis_interpretation = "As imagens são pouco semelhantes."

        # Resultados da análise
        result_text = [
            html.H4(f'Mean Squared Error: {mse_value:.4f}'),
            html.H4(f'Structural Similarity Index: {ssim_value:.4f}'),
            html.H4(f'Correlação de Pearson: {pearson_corr:.4f}'),
            html.H4(f'Análise: {analysis_interpretation}')
        ]

        # Exibir as imagens usando Plotly Express
        fig1 = px.imshow(img1_array, title="Imagem 1")
        fig2 = px.imshow(img2_resized_array, title="Imagem 2")

        return result_text, fig1, fig2

    return html.Div("Carregue as imagens e clique no botão para analisar"), {}, {}


if __name__ == '__main__':
    app.run_server(debug=False)
