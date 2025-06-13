# pages/2_🔎_Análisis_de_Cartera.py

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuración de la página ---
st.set_page_config(page_title="Análisis de Cartera", layout="wide")
st.title(" Análisis de Cartera de Clientes")
st.write(
    "Utiliza este dashboard para obtener una vista general de tu portafolio de clientes, "
    "analizar distribuciones clave y buscar clientes individuales."
)

if 'df' not in st.session_state:
    st.warning("Por favor, carga tus datos en la página de 'Inicio' primero.")
    st.stop()

# Usamos el dataframe original para el análisis
df_original = st.session_state['df']

# --- Dashboard de Análisis de Portafolio ---
st.header("Dashboard General del Portafolio")

# Métricas clave
col1, col2, col3 = st.columns(3)
total_clientes = len(df_original)
saldo_total = df_original['Saldo_total'].sum()
utilizacion_promedio = df_original['Utilizacion'].mean()

col1.metric("Número Total de Clientes", f"{total_clientes:,}")
col2.metric("Saldo Total en Cartera", f"${saldo_total:,.2f} MXN")
col3.metric("Utilización Promedio", f"{utilizacion_promedio:.2%}")

st.divider()

# --- Visualizaciones ---
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("Distribución de Clientes por Socio")
    socio_counts = df_original['Socio'].value_counts().reset_index()
    socio_counts.columns = ['Socio', 'Número de Clientes']
    fig_socio = px.pie(
        socio_counts,
        names='Socio',
        values='Número de Clientes',
        title='Proporción de Clientes por Socio',
        hole=0.3
    )
    st.plotly_chart(fig_socio, use_container_width=True)

with col_viz2:
    st.subheader("Distribución de Saldos en la Cartera")
    fig_saldo = px.histogram(
        df_original,
        x='Saldo_total',
        nbins=50,
        title='Frecuencia de Saldos Totales',
        labels={'Saldo_total': 'Saldo Total (MXN)'}
    )
    st.plotly_chart(fig_saldo, use_container_width=True)

st.divider()

# --- Búsqueda de Clientes ---
st.header("Búsqueda de Clientes Individuales")
st.write("Busca un cliente por su número de índice (ID de Cliente) para ver su perfil completo.")

# Crear un identificador único si no existe (usando el índice)
if 'ID_Cliente' not in df_original.columns:
    df_original.insert(0, 'ID_Cliente', range(1, len(df_original) + 1))

# Widget de búsqueda
cliente_id_seleccionado = st.selectbox(
    'Selecciona un ID de Cliente:',
    df_original['ID_Cliente']
)

if cliente_id_seleccionado:
    # Filtrar el dataframe para obtener los datos del cliente
    datos_cliente = df_original[df_original['ID_Cliente'] == cliente_id_seleccionado].iloc[0]
    
    st.subheader(f"Perfil del Cliente: {cliente_id_seleccionado}")
    
    # Mostrar los datos del cliente en un formato más legible
    perfil = {
        "Información Principal": {
            "Socio": datos_cliente.get('Socio'),
            "Producto": datos_cliente.get('Producto'),
            "Fecha de Aprobación": datos_cliente.get('Fecha_aprobacion'),
            "Fecha de Activación": datos_cliente.get('Fecha_activacion'),
        },
        "Estado Financiero Actual": {
            "Saldo Total": f"${datos_cliente.get('Saldo_total', 0):,.2f}",
            "Saldo del Mes": f"${datos_cliente.get('Saldo_Mes', 0):,.2f}",
            "Límite de Crédito": f"${datos_cliente.get('Limite_credito', 0):,.2f}",
            "Utilización": f"{datos_cliente.get('Utilizacion', 0):.2%}",
        },
        "Comportamiento de Pago": {
            "Score de Pago": datos_cliente.get('Score_pago'),
            "Ciclos de Atraso (Mes 1)": datos_cliente.get('Ciclo_atraso_M1'),
            "Canal de Pago (Mes 1)": datos_cliente.get('Canal_Pago_M1'),
        }
    }

    # Mostrar el perfil en columnas
    col_perfil1, col_perfil2 = st.columns(2)
    with col_perfil1:
        for seccion, datos in perfil.items():
            if seccion != "Comportamiento de Pago":
                st.write(f"**{seccion}**")
                st.json(datos, expanded=True)
    with col_perfil2:
        st.write(f"**Comportamiento de Pago**")
        st.json(perfil["Comportamiento de Pago"], expanded=True)