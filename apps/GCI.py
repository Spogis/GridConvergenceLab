import pandas as pd
import numpy as np


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

        # Arredondar os valores para 4 casas decimais
        p_values = np.round(p_values, 4)
        gci_fine_values = np.round(gci_fine_values, 4)
        gci_medium_values = np.round(gci_medium_values, 4)

    return p_values, gci_medium_values, gci_fine_values
