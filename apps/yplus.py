import numpy as np
import math
from decimal import Decimal

def yplus(density=1000, viscosity=0.001, desired_yplus=30, growth_rate=1.2,
          freestream_velocity=1, characteristic_length=1, diameter=1, rpm=420, option='Free'):

    Tip_Speed = math.pi * diameter * rpm / 60
    if option != 'Free':
        freestream_velocity = Tip_Speed
        characteristic_length = diameter
        print('Impeller')

    Reynolds = density * freestream_velocity * characteristic_length / viscosity
    Cf = (2.0 * np.log10(Reynolds) - 0.65) ** (-2.3)
    Tw = Cf * (1 / 2) * density * (freestream_velocity ** 2)
    Uplus = (Tw / density) ** (1 / 2)

    # First Layer Thickness
    DeltaY = desired_yplus * viscosity / (Uplus * density)

    # Boundary Layer Thickness
    Boundary_Layer_Thickness = 0.035 * characteristic_length * Reynolds ** (-1 / 7)

    # Number of Layers
    Number_Of_Layers = round(np.log(((Boundary_Layer_Thickness * (growth_rate - 1)) / DeltaY) + 1) / np.log(growth_rate), 0)

    return (Reynolds, DeltaY, Boundary_Layer_Thickness, Number_Of_Layers)