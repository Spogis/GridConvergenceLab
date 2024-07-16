import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import plotly.graph_objs as go
import base64
import io

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Upload de Arquivos Excel e Análise de Dados"),
    html.Div([
        dcc.Upload(
            id='upload-data-1',
            children=html.Div([
                'Arraste e solte ou ',
                html.A('selecione o arquivo 1')
            ]),
            style={
                'width': '48%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Div(id='output-file-upload-1')
    ]),
    html.Div([
        dcc.Upload(
            id='upload-data-2',
            children=html.Div([
                'Arraste e solte ou ',
                html.A('selecione o arquivo 2')
            ]),
            style={
                'width': '48%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            }
        ),
        html.Div(id='output-file-upload-2')
    ]),
    html.Button('Gerar Análise', id='analyze-button', n_clicks=0),
    html.Div(id='output-analysis'),
    dcc.Graph(id='output-graph')
])

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

if __name__ == '__main__':
    app.run_server(debug=True)
