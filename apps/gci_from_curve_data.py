import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def gci_from_curve_data(df_coarse, df_medium, df_fine, N_Splits):

    # Interpolação
    x_common = np.linspace(min(df_coarse['x'].min(), df_medium['x'].min(), df_fine['x'].min()),
                           max(df_coarse['x'].max(), df_medium['x'].max(), df_fine['x'].max()),
                           N_Splits)

    spline_coarse = interp1d(df_coarse['x'], df_coarse['y'], kind='cubic', fill_value="extrapolate")
    spline_medium = interp1d(df_medium['x'], df_medium['y'], kind='cubic', fill_value="extrapolate")
    spline_fine = interp1d(df_fine['x'], df_fine['y'], kind='cubic', fill_value="extrapolate")

    y_coarse_interp = spline_coarse(x_common)
    y_medium_interp = spline_medium(x_common)
    y_fine_interp = spline_fine(x_common)

    phi_names = np.array([f'phi_{x:.4f}' for x in x_common])
    df_coarse_interp = pd.DataFrame({'x': phi_names, 'y': y_coarse_interp})
    df_medium_interp = pd.DataFrame({'x': phi_names, 'y': y_medium_interp})
    df_fine_interp = pd.DataFrame({'x': phi_names, 'y': y_fine_interp})

    df_curve_for_gci = pd.DataFrame({'variable': phi_names,
                                     'coarse': y_coarse_interp,
                                     'medium': y_medium_interp,
                                     'fine': y_fine_interp,
                                     'x': x_common})

    df_curve_for_gci.to_excel('data/curve_for_gci.xlsx', index=False)
