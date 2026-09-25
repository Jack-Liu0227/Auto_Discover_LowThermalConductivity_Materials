#!/user/bin/env python3
# -*- coding: utf-8 -*-
import math
from scipy.constants import h, k
import pandas as pd
import numpy as np




def cal_Debye_T(df):
    """
    Calculate Debye temperature and related parameters
    """
    B="Bulk modulus (GPa)"
    G="Shear modulus (GPa)"
    Va = "Speed of sound (m s-1)"
    density = "Density (g cm-3)"
    L="Sound velocity of the longitude wave (m s-1)"
    T="Sound velocity of the transverse wave (m s-1)"
    V = "Volume (Å3)"
    Debye="Acoustic Debye Temperature (K)"
    Vl = ((df[B] + 4 * df[G] / 3) / df[density]) ** 0.5
    Vt = (df[G]/ df[density]) ** 0.5
    Vs = ((1 / pow(Vl, 3) + 2 / pow(Vt, 3)) / 3) ** (-1 / 3)
    df[L]=Vl*1000
    df[T]=Vt*1000
    df[Va] = Vs * 1000
    df[Debye] = h / k * np.power(3 / (4 * math.pi * np.array(df[V])), 1 / 3) * np.array(df[Va])*math.pow(10,10)
    return df

def cal_gamma(df, custom_gamma=None):
    """
    Calculate Grüneisen parameter and Poisson ratio
    If custom_gamma is provided, use that value directly
    """
    gamma_df = df.copy()
    L = gamma_df['Sound velocity of the longitude wave (m s-1)']
    T = gamma_df['Sound velocity of the transverse wave (m s-1)']
    a = L / T
    gamma_df['Poisson ratio'] = (pow(a, 2) - 2) / (2*pow(a, 2) - 2)
    
    if custom_gamma is not None:
        # If custom value is provided, use it directly
        gamma_df['Grüneisen parameter'] = custom_gamma
    else:
        gamma_df['Grüneisen parameter'] = 3 * (1 + gamma_df['Poisson ratio']) / (2 * (2 - 3 * gamma_df['Poisson ratio']))
    
    return gamma_df

def cal_A(df, n, custom_gamma=None):
    """
    Calculate A value, optionally using custom Grüneisen parameter
    """
    gamma = "Grüneisen parameter"
    A="A"
    if custom_gamma is not None:
        gamma_value = custom_gamma
    else:
        gamma_value = df[gamma]
    
    if n==1:
        df[A]=2.43e-8/(1-0.514/gamma_value+0.228/pow(gamma_value,2))
    else:
        df[A] = 1 / (1 +1 / gamma_value + 8.3e5 / pow(gamma_value, 2.4))
    return df

def cal_K_Slack(df):
    A="A"
    M="Total Atomic Mass (amu)"
    N = "Number of Atoms"
    Debye = "Acoustic Debye Temperature (K)"
    V="Volume (Å3)"
    gamma = "Grüneisen parameter"
    T=300
    K_Slack="Kappa_Slack (W m-1 K-1)"
    df[K_Slack]=df[A]*df[M]*pow(df[V],1/3)*pow(df[Debye],3)/(pow(df[gamma],2)*T*df[N])*100
    return(df)

def by_MTP(df):
    """
    Calculate thermal conductivity using MTP method
    """
    import numpy as np
    K_df = df.copy()
    
    # Ensure all required columns exist
    required_columns = ['Grüneisen parameter', 'Shear modulus (GPa)', 
                       'Volume (Å3)', 'Number of Atoms', 'Speed of sound (m s-1)']
    for col in required_columns:
        if col not in K_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Get required values and convert to numpy arrays
    gamma = K_df['Grüneisen parameter'].astype(float).values
    G = K_df['Shear modulus (GPa)'].astype(float).values * 1e9  # Convert to Pa
    V = K_df['Volume (Å3)'].astype(float).values * 1e-30  # Convert to m³
    N = K_df['Number of Atoms'].astype(float).values
    vs = K_df['Speed of sound (m s-1)'].astype(float).values
    T = 300  # Temperature set to 300K
    
    # Calculate thermal conductivity (W/mK)
    V_term = np.power(V, 1/3)
    exp_term = np.exp(-gamma)
    kappa = G * vs * V_term / (N * T) * exp_term
    
    # Ensure results are finite values
    kappa = np.where(np.isfinite(kappa), kappa, 0)
    K_df['Kappa_cal (W m-1 K-1)'] = kappa
    
    return K_df

if __name__=="__main__":
    pass
