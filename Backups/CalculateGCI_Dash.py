import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import io
import base64
import os

# Inicializando o aplicativo Dash
app = dash.Dash(__name__)


# Função para calcular GCI para múltiplas variáveis
def calculate_gci_multiple_variables(results_coarse, results_medium, results_fine, r=2):
    num_vars = len(results_coarse)
    p_values = np.zeros(num_vars)
    gci_fine_values = np.zeros(num_vars)
    gci_medium_values = np.zeros(num_vars)

    for i in range(num_vars):
        # Calcular a taxa de convergência aparente (p) para cada variável
        p_values[i] = np.log((results_coarse[i] - results_medium[i]) / (results_medium[i] - results_fine[i])) / np.log(
            r)

        # Calcular o fator de segurança
        Fs = 1.25

        # Calcular o GCI para a malha média e fina
        gci_fine_values[i] = Fs * np.abs((results_fine[i] - results_medium[i]) / results_fine[i]) / (
                    r ** p_values[i] - 1)
        gci_medium_values[i] = Fs * np.abs((results_medium[i] - results_coarse[i]) / results_medium[i]) / (
                    r ** p_values[i] - 1)

    return p_values, gci_fine_values, gci_medium_values


# Layout do aplicativo
app.layout = html.Div([
    html.H1("Calculadora de GCI"),
    dash_table.DataTable(
        id='editable-table',
        columns=[
            {'name': 'Variável', 'id': 'variable', 'type': 'text'},
            {'name': 'Malha Grosseira', 'id': 'coarse', 'type': 'numeric'},
            {'name': 'Malha Média', 'id': 'medium', 'type': 'numeric'},
            {'name': 'Malha Fina', 'id': 'fine', 'type': 'numeric'}
        ],
        data=[{'variable': f'Var{i + 1}', 'coarse': 0, 'medium': 0, 'fine': 0} for i in range(3)],
        editable=True,
        row_deletable=True
    ),
    html.Br(),
    html.Button('Adicionar Linha', id='add-row-button', n_clicks=0),
    html.Button('Salvar Tabela', id='save-button', n_clicks=0),

    html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Carregar Tabela'),
        multiple=False
    ),

    html.Br(),
    html.Button('Calcular GCI', id='calculate-button', n_clicks=0),
    html.Button('Baixar Resultados GCI', id='download-gci-button', n_clicks=0),
    html.Br(),

    dash_table.DataTable(id='gci-results-table'),
    dcc.Download(id="download-table"),
    dcc.Download(id="download-gci")
])


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
    prevent_initial_call=True
)
def calculate_gci(n_clicks, rows):
    if n_clicks > 0:
        variables = [row['variable'] for row in rows]
        results_coarse = [float(row['coarse']) for row in rows]
        results_medium = [float(row['medium']) for row in rows]
        results_fine = [float(row['fine']) for row in rows]

        r = 2  # Fator de refinamento

        p_values, gci_fine_values, gci_medium_values = calculate_gci_multiple_variables(results_coarse, results_medium,
                                                                                        results_fine, r)

        df_results = pd.DataFrame({
            'Variável': variables,
            'p': p_values,
            'GCI Malha Fina': gci_fine_values,
            'GCI Malha Média': gci_medium_values
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
    if not os.path.exists('../setups'):
        os.makedirs('../setups')
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


# Rodar o aplicativo
if __name__ == '__main__':
    app.run_server(debug=True)
