import pandas as pd
import numpy as np
from pyGCS import GCI, GCS


def calculate_gci_multiple_variables(values_coarse, values_medium, values_fine,
                                     nodes_coarse, nodes_medium, nodes_fine, volume,
                                     mesh_type):


    # Listas para armazenar os resultados
    GCI_fine_list = []
    GCI_medium_list = []
    p_list = []
    GCI_asymptotic_list = []
    phi_extrapolated_list = []
    r_fine_list = []
    r_medium_list = []
    e_fine_list = []
    e_medium_list = []

    for value_coarse, value_medium, value_fine in zip(values_coarse, values_medium, values_fine):

        gci = GCI(dimension=mesh_type, simulation_order=2, volume=volume,
                  cells=[nodes_fine, nodes_medium, nodes_coarse],
                  solution=[value_fine, value_medium, value_coarse])

        p = gci.get('apparent_order')
        GCI_asymptotic = gci.get('asymptotic_gci')
        GCI_fine, GCI_medium = gci.get('gci')
        phi_extrapolated = gci.get('extrapolated_value')
        r_fine, r_medium =  gci.get('refinement_ratio')
        e_fine, e_medium = gci.get('relative_error')

        # Armazenar os resultados
        GCI_fine_list.append(GCI_fine)
        GCI_medium_list.append(GCI_medium)
        p_list.append(p)
        GCI_asymptotic_list.append(GCI_asymptotic)
        phi_extrapolated_list.append(phi_extrapolated)
        r_fine_list.append(r_fine)
        r_medium_list.append(r_medium)
        e_fine_list.append(e_fine)
        e_medium_list.append(e_medium)

    return (p_list, GCI_medium_list, GCI_fine_list, GCI_asymptotic_list, phi_extrapolated_list,
            r_fine_list, r_medium_list)

