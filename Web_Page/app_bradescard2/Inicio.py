# 1_🏠_Inicio.py

import streamlit as st
import pandas as pd

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis Predictivo de Clientes",
    page_icon="🤖",
    layout="wide"
)

# --- Título y descripción ---
st.title("Bradescard: Análisis de Portafolio de Clientes")
st.write(
    "Bienvenido a la herramienta de análisis de cartera. Esta plataforma te permite "
    "analizar datos de clientes para identificar patrones y segmentar tu portafolio. "
    "Puedes usar nuestros datos de demostración o subir tu propio archivo para comenzar."
)

st.divider()

# --- Carga de Datos ---
st.header("Paso 1: Carga tus Datos")

# Nota sobre el límite de carga
st.info(
    "**Nota sobre el tamaño de los archivos:** Streamlit tiene un límite de carga de 200 MB por defecto. "
)

# Opción para elegir la fuente de datos
source_option = st.radio(
    "Elige una fuente de datos:",
    ("Usar datos de demostración (COLL_TEC_CONSOLIDADO.txt)", "Subir mi propio archivo (.csv o .txt)"),
    captions=[
        "Analiza un set de datos precargado para explorar las funcionalidades.",
        "Sube tu propio archivo para un análisis personalizado (hasta 1 GB)."
    ]
)

uploaded_file = None
if "Usar datos de demostración" in source_option:
    try:
        df = pd.read_csv("data/COLL_TEC_CONSOLIDADO.txt", delimiter=",", encoding="latin-1", low_memory=False)
        st.session_state['df'] = df
        st.success("Datos de demostración cargados correctamente.")
    except FileNotFoundError:
        st.error(
            "Error: No se encontró el archivo 'data/COLL_TEC_CONSOLIDADO.txt'. "
            "Asegúrate de que el archivo esté en la carpeta 'data'."
        )
        st.stop()

else:
    # Actualizado para aceptar CSV y TXT
    uploaded_file = st.file_uploader(
        "Sube tu archivo",
        type=['csv', 'txt'],
        help="El archivo debe ser formato CSV o TXT delimitado por comas."
    )
    if uploaded_file is not None:
        try:
            # Asumimos delimitador por comas para ambos tipos de archivo, como en el notebook
            df = pd.read_csv(uploaded_file, delimiter=",", encoding="latin-1", low_memory=False)
            st.session_state['df'] = df
            st.success("Archivo cargado exitosamente.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo. Asegúrate de que sea un CSV o TXT válido y delimitado por comas. Error: {e}")
            st.stop()

# --- Vista Previa de los Datos ---
if 'df' in st.session_state:
    st.subheader("Vista Previa de los Datos Cargados")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.info("Datos listos. Explora las páginas de 'Análisis de Cartera' y 'Análisis de Riesgo' en la barra lateral.")