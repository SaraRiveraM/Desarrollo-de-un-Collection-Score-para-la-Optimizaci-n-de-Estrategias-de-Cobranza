# pages/3_ρί_Análisis_de_Riesgo.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Análisis de Riesgo", layout="wide")

st.title("Análisis de Riesgo y Segmentación de Clientes")
st.write(
    "Este módulo te permite analizar la cartera según el 'Score de Pago' y también mediante "
    "algoritmos de Machine Learning para encontrar agrupaciones naturales (clústeres)."
)

if 'df' not in st.session_state:
    st.warning("Por favor, carga tus datos en la página de 'Inicio' primero.")
    st.stop()

df = st.session_state['df'].copy()

# --- Sección 1: Segmentación por Reglas de Negocio (Score de Pago) ---
st.header("Segmentación por Score de Pago")

def asignar_riesgo(score):
    if score >= 9: return 'Bajo Riesgo'
    elif score >= 5: return 'Riesgo Medio'
    elif score > 0: return 'Alto Riesgo'
    else: return 'Sin Información'

df['Nivel_de_Riesgo'] = df['Score_pago'].apply(asignar_riesgo)

riesgo_counts = df['Nivel_de_Riesgo'].value_counts()
fig_riesgo = px.bar(
    riesgo_counts, y=riesgo_counts.values, x=riesgo_counts.index,
    title='Distribución de Clientes por Nivel de Riesgo (Score)',
    labels={'y': 'Número de Clientes', 'x': 'Nivel de Riesgo'},
    color=riesgo_counts.index,
    color_discrete_map={'Alto Riesgo': '#E74C3C', 'Riesgo Medio': '#F39C12', 'Bajo Riesgo': '#2ECC71', 'Sin Información': '#BDC3C7'}
)
st.plotly_chart(fig_riesgo, use_container_width=True)

analisis_riesgo = df.groupby('Nivel_de_Riesgo').agg(
    Saldo_Total_Promedio=('Saldo_total', 'mean'),
    Utilizacion_Promedio=('Utilizacion', 'mean'),
    Numero_de_Clientes=('Socio', 'count')
).reset_index()

st.write("Análisis Comparativo por Segmento de Score:")
st.dataframe(analisis_riesgo.style.format({
    "Saldo_Total_Promedio": "${:,.2f}", "Utilizacion_Promedio": "{:.2%}"
}), use_container_width=True)

st.divider()

# --- Sección 2: Segmentación por Machine Learning (Clustering) ---
st.header("Segmentación por Comportamiento (K-Means Clustering)")
st.write("Usa Machine Learning para agrupar a tus clientes en 3 clústeres basados en su comportamiento financiero general.")

if st.button("Ejecutar Análisis de Clustering", type="primary"):
    with st.spinner("Procesando datos y calculando clústeres..."):
        # 1. Preparar datos (versión simplificada de tu pipeline para clustering)
        df_proc = df.copy()
        numeric_cols = df_proc.select_dtypes(include=['number']).columns
        df_proc[numeric_cols] = df_proc[numeric_cols].fillna(0)
        
        # Eliminar columnas no relevantes para el clustering de comportamiento
        cols_to_drop = ['Variable_objetivo', 'ID_Cliente', 'Score_pago', 'Nivel_de_Riesgo']
        df_proc = df_proc.drop(columns=[col for col in cols_to_drop if col in df_proc.columns], errors='ignore')
        
        # Seleccionar solo las columnas numéricas para el modelo
        df_numeric = df_proc.select_dtypes(include=['number'])

        # 2. Escalar los datos
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)

        # 3. Aplicar KMeans con 3 clústeres
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        df['Cluster'] = kmeans.fit_predict(df_scaled)
        st.session_state['df_clustered'] = df # Guardar para no recalcular
        st.success("¡Clustering completado! Se han identificado 3 segmentos.")

if 'df_clustered' in st.session_state:
    df_clustered = st.session_state['df_clustered']
    
    # 4. Visualizar los clústeres con PCA
    st.subheader("Visualización de los Clústeres")
    
    # Reducir dimensionalidad a 2 componentes para graficar
    pca = PCA(n_components=2)
    df_numeric_scaled = StandardScaler().fit_transform(df_clustered.select_dtypes(include=['number']).fillna(0))
    components = pca.fit_transform(df_numeric_scaled)
    
    df_clustered['pca_x'] = components[:, 0]
    df_clustered['pca_y'] = components[:, 1]

    fig_cluster = px.scatter(
        df_clustered,
        x='pca_x',
        y='pca_y',
        color='Cluster',
        title='Segmentos de Clientes (Visualización con PCA)',
        labels={'pca_x': 'Componente Principal 1', 'pca_y': 'Componente Principal 2'},
        hover_data=['Saldo_total', 'Utilizacion', 'Score_pago']
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # 5. Analizar los clústeres
    st.subheader("Análisis Comparativo por Clúster")
    cluster_analysis = df_clustered.groupby('Cluster').agg(
        Saldo_Total_Promedio=('Saldo_total', 'mean'),
        Utilizacion_Promedio=('Utilizacion', 'mean'),
        Score_Pago_Promedio=('Score_pago', 'mean'),
        Numero_de_Clientes=('Socio', 'count')
    ).reset_index()

    st.dataframe(cluster_analysis.style.format({
        "Saldo_Total_Promedio": "${:,.2f}",
        "Utilizacion_Promedio": "{:.2%}",
        "Score_Pago_Promedio": "{:.1f}"
    }), use_container_width=True)
    
    st.info(
        "**¿Cómo interpretar los clústeres?**\n\n"
        "Cada clúster (0, 1, 2) representa un grupo de clientes con características financieras similares. "
        "Analiza la tabla de arriba para entender el 'perfil' de cada grupo. Por ejemplo, un clúster podría "
        "agrupar a clientes de 'alto saldo y alta utilización', mientras que otro podría ser de 'bajo saldo y bajo riesgo'."
    )