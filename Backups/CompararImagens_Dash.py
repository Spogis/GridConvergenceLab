import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
import base64
import io
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

# Inicializar o aplicativo Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Upload de Imagens e Análise de Similaridade"),
    html.Div([
        dcc.Upload(
            id='upload-image-1',
            children=html.Div([
                'Arraste e solte ou ',
                html.A('selecione a imagem 1')
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
        html.Div(id='output-image-upload-1')
    ]),
    html.Div([
        dcc.Upload(
            id='upload-image-2',
            children=html.Div([
                'Arraste e solte ou ',
                html.A('selecione a imagem 2')
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
        html.Div(id='output-image-upload-2')
    ]),
    html.Button('Gerar Análise', id='analyze-button', n_clicks=0),
    html.Div(id='output-analysis'),
    dcc.Graph(id='output-graph-1'),
    dcc.Graph(id='output-graph-2')
])

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
    [Output('output-analysis', 'children'),
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

if __name__ == '__main__':
    app.run_server(debug=True)
