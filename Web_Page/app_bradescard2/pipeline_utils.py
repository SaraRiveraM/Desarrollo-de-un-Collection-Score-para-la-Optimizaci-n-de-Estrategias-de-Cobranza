# pipeline_utils.py
# This file contains the custom function and class definitions required by the ML pipeline.
# It is based on the logic developed in the pipeline.ipynb notebook.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def limpieza_sin_categoricas(df_copy):
    """
    Performs data cleaning on the dataframe. This function handles date conversion,
    imputation, feature engineering, and dropping of unnecessary columns based on
    the logic from the development notebook.
   
    """
    df_copy = df_copy.copy()

    ## Convertir columnas que empiezan con "Fecha" o "Prox" a tipo fecha
    for column in df_copy.columns:
        if column.startswith('Fecha') or column.startswith('Prox'):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')

    # Agregar las 6 columnas con valor inicial en 1
    for i in range(1, 7):
        df_copy[f'Activo_M{i}'] = 1

    meses = range(1, 7)
    for j in meses:
        columnas_check = [f'Saldo_total_M{k}' for k in range(j, 7)]
        columnas_fill = [
            f'Ciclo_atraso_M{j}', f'Pago_M{j}', f'Fecha_pago_M{j}', f'Utilizacion_M{j}',
            f'Fecha_corte_M{j}', f'Fecha_limite_pago_M{j}'
        ]
        mask_nulos = df_copy[columnas_check].isnull().all(axis=1) #
        df_copy.loc[mask_nulos, columnas_fill] = df_copy.loc[mask_nulos, columnas_fill].fillna(0) #
        df_copy.loc[mask_nulos, f'Activo_M{j}'] = 0 #

    if 'Fecha_prox_corte_M1' in df_copy.columns:
        mask_fecha_prox = df_copy[[f'Saldo_total_M{k}' for k in meses]].isnull().all(axis=1) #
        df_copy.loc[mask_fecha_prox, 'Fecha_prox_corte_M1'] = 0 #

    # Drop columnas 'Behavior'
    df_copy = df_copy.drop(df_copy.filter(regex='Behavior').columns, axis=1) #

    # Canal_Pago y Canal_Pago_M1-M6
    columnas_canal_pago = [f'Canal_Pago_M{j}' for j in meses]
    df_copy[columnas_canal_pago] = df_copy[columnas_canal_pago].fillna('Desconocido') #
    df_copy['Moda_Canal_Pago'] = df_copy[columnas_canal_pago].mode(axis=1)[0] #
    df_copy['Canal_Pago'].fillna(df_copy['Moda_Canal_Pago'], inplace=True) #
    df_copy.drop(columns=['Moda_Canal_Pago'], inplace=True) #

    # Eliminar columnas de Fecha_prox_corte_M2-M6 si coinciden con fechas corte previas
    mask_fecha_igual = False
    for i in range(2, 7):
        if f'Fecha_prox_corte_M{i}' in df_copy.columns and f'Fecha_corte_M{i-1}' in df_copy.columns:
            mask_fecha_igual |= (df_copy[f'Fecha_prox_corte_M{i}'] == df_copy[f'Fecha_corte_M{i-1}'])

    df_copy = df_copy[mask_fecha_igual]
    columnas_a_eliminar = [f'Fecha_prox_corte_M{i}' for i in range(2, 7)]
    df_copy.drop(columns=[col for col in columnas_a_eliminar if col in df_copy.columns], inplace=True) #

    # Convertir fechas a días desde fecha base
    fecha_base = pd.to_datetime('01/01/01')
    for i in range(1, 7):
        for col in [f'Fecha_corte_M{i}', f'Fecha_limite_pago_M{i}', f'Fecha_pago_M{i}']:
            if col in df_copy.columns:
                mask_zeros = df_copy[col] == 0
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                df_copy[col] = (df_copy[col] - fecha_base).dt.days #
                df_copy.loc[mask_zeros, col] = 0

    # Saldo_total_M1, Saldo_Mes_M1, Pago_minimo_M1
    mask_nulos_m1 = df_copy[['Saldo_total_M1', 'Saldo_Mes_M1', 'Pago_minimo_M1']].isnull().all(axis=1) #
    columnas_m1 = df_copy.columns[df_copy.columns.str.contains('M1')]
    df_copy.loc[mask_nulos_m1, columnas_m1] = df_copy.loc[mask_nulos_m1, columnas_m1].fillna(0) #

    # Eliminar columnas innecesarias
    df_copy.drop(df_copy.filter(regex='Genero').columns, axis=1, inplace=True) #
    df_copy.drop(df_copy.filter(regex='Fecha_pago').columns, axis=1, inplace=True) #
    for i in range(1, 7):
        df_copy.drop(df_copy.filter(regex=f'Pago_M{i}').columns, axis=1, inplace=True)
    df_copy.drop(columns=['Pago'], errors='ignore', inplace=True)

    # Crear columnas Deuda_M1 a Deuda_M6
    for i in range(1, 7):
        df_copy[f'Deuda_M{i}'] = df_copy['Limite_credito'] * df_copy[f'Utilizacion_M{i}'] #

    # Eliminar filas con datos faltantes en columnas críticas
    columnas_a_verificar = sum([[f'Saldo_total_M{i}', f'Saldo_Mes_M{i}', f'Pago_minimo_M{i}'] for i in range(1, 7)], [])
    df_copy = df_copy.dropna(subset=columnas_a_verificar) #
    df_copy.dropna(inplace=True) #

    # Convertir columnas tipo fecha restantes a días desde fecha_base
    for column in df_copy.columns:
        if column.startswith('Fecha') and not column.endswith(('M1', 'M2', 'M3', 'M4', 'M5', 'M6')):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')
        if column.startswith('Prox'):
            df_copy[column] = pd.to_datetime(df_copy[column], errors='coerce')

    df_dates = df_copy.select_dtypes(include=['datetime64[ns]'])
    for col in df_dates.columns:
        mask_zeros = df_copy[col] == 0
        df_copy[col] = (df_copy[col] - fecha_base).dt.days #
        df_copy.loc[mask_zeros, col] = 0

    # Especial: Fecha_prox_corte_M1
    if 'Fecha_prox_corte_M1' in df_copy.columns:
        mask_zeros_2 = df_copy['Fecha_prox_corte_M1'] == 0
        df_copy['Fecha_prox_corte_M1'] = pd.to_datetime(df_copy['Fecha_prox_corte_M1'], errors='coerce')
        df_copy['Fecha_prox_corte_M1'] = (df_copy['Fecha_prox_corte_M1'] - fecha_base).dt.days #
        df_copy['Fecha_prox_corte_M1'] = df_copy['Fecha_prox_corte_M1'].fillna(0).astype(int)

    return df_copy


class Preprocesador(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply OneHotEncoding to categorical features and
    StandardScaling to numerical features.
    """
    def __init__(self):
        self.ohe = None
        self.scaler = None
        self.categorical_cols = None
        self.numeric_cols = None

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist() #
        self.numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist() #

        self.ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False) #
        self.ohe.fit(X[self.categorical_cols])

        self.scaler = StandardScaler() #
        self.scaler.fit(X[self.numeric_cols])

        return self

    def transform(self, X):
        X = X.copy()

        X_cat = pd.DataFrame(self.ohe.transform(X[self.categorical_cols]),
                             columns=self.ohe.get_feature_names_out(self.categorical_cols),
                             index=X.index)

        X_num = pd.DataFrame(self.scaler.transform(X[self.numeric_cols]),
                             columns=self.numeric_cols,
                             index=X.index)

        return pd.concat([X_num, X_cat], axis=1)


class PCAWithTarget(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply PCA to the features and then re-attach the
    target variable to the resulting DataFrame.
    """
    def __init__(self, target_column='Variable_objetivo', n_components=70):
        self.target_column = target_column
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components) #

    def fit(self, X, y=None):
        X_features = X.drop(columns=[self.target_column])
        self.pca.fit(X_features)
        return self

    def transform(self, X):
        X_features = X.drop(columns=[self.target_column]) #
        y_target = X[self.target_column].reset_index(drop=True) #

        X_pca = self.pca.transform(X_features)
        df_pca = pd.DataFrame(X_pca, columns=[f'PCA_{i+1}' for i in range(self.n_components)]) #
        df_pca[self.target_column] = y_target.values #

        return df_pca