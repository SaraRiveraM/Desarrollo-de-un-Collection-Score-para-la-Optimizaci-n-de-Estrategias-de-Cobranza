# pages/4_🧠_Modelo_Predictivo.py

import streamlit as st
import pandas as pd
import joblib
import sys

# Importar las clases y funciones personalizadas del script de utilidades
from pipeline_utils import Preprocesador, PCAWithTarget, limpieza_sin_categoricas

# ------------------- INYECCIÓN DE DEPENDENCIAS (FIX) -------------------
# Se inyectan los objetos personalizados en el espacio de nombres de __main__
# para que joblib/pickle pueda encontrarlos al cargar el pipeline.
sys.modules['__main__'].limpieza_sin_categoricas = limpieza_sin_categoricas
sys.modules['__main__'].Preprocesador = Preprocesador
sys.modules['__main__'].PCAWithTarget = PCAWithTarget
# -----------------------------------------------------------------

st.set_page_config(page_title="Modelo Predictivo", layout="wide")
st.title("Modelo Predictivo de Comportamiento")

# --- Cargar Datos y Artefactos del Modelo ---
if 'df' not in st.session_state:
    st.warning("Por favor, carga tus datos en la página de 'Inicio' primero.")
    st.stop()

try:
    @st.cache_resource
    def load_artifacts():
        pipeline = joblib.load('fitted_pipeline.pkl')
        model = joblib.load('random_forest_model.pkl')
        return pipeline, model
    
    pipeline, model = load_artifacts()
    st.success("Modelo predictivo y pipeline de procesamiento cargados exitosamente.")

except Exception as e:
    st.error(f"Ocurrió un error al cargar los artefactos del modelo: {e}")
    st.stop()


# --- Sección 1: Predicción para toda la Cartera ---
st.header("1. Ejecutar Predicciones para toda la Cartera")
st.write("Usa este botón para aplicar el modelo a todos los clientes del archivo que cumplen con los criterios de limpieza de datos.")

if st.button("Predecir para toda la Cartera", type="primary"):
    with st.spinner("Procesando datos y generando predicciones..."):
        df_predict = st.session_state['df'].copy()

        if 'Variable_objetivo' not in df_predict.columns:
            df_predict['Variable_objetivo'] = 0

        try:
            transformed_data = pipeline.transform(df_predict)
            predictions = model.predict(transformed_data)
            
            df_results = st.session_state['df'].loc[transformed_data.index].copy()
            df_results['Prediccion_Modelo'] = predictions
            
            st.subheader("Resultados de la Predicción en Lote")
            st.write(f"El modelo se pudo ejecutar en {len(df_results)} de {len(st.session_state['df'])} clientes.")
            st.dataframe(df_results[['Socio', 'Producto', 'Saldo_total', 'Score_pago', 'Prediccion_Modelo']])
        
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción en lote: {e}")

st.divider()

# --- Sección 2: Predicción para un Cliente Individual ---
st.header("2. Clasificar un Cliente Individual")
st.write("Selecciona un cliente del archivo cargado para obtener una clasificación individual.")

# Asegurarse de que la columna ID_Cliente exista para la selección
df_main = st.session_state['df']
if 'ID_Cliente' not in df_main.columns:
    df_main.insert(0, 'ID_Cliente', range(1, len(df_main) + 1))

# Widget para seleccionar el cliente
cliente_id_seleccionado = st.selectbox(
    'Selecciona un ID de Cliente:',
    options=df_main['ID_Cliente'].unique(),
    help="Elige un cliente del archivo que cargaste en la página de Inicio."
)

if st.button("Clasificar Cliente Seleccionado"):
    if cliente_id_seleccionado:
        with st.spinner(f"Clasificando al cliente {cliente_id_seleccionado}..."):
            # Obtener la fila de datos para el cliente seleccionado
            datos_cliente = df_main[df_main['ID_Cliente'] == cliente_id_seleccionado].copy()

            # Añadir la columna dummy de 'Variable_objetivo' si no existe
            if 'Variable_objetivo' not in datos_cliente.columns:
                datos_cliente['Variable_objetivo'] = 0

            try:
                # El pipeline puede eliminar al cliente si no cumple los criterios.
                # Debemos verificar si sobrevivió.
                transformed_cliente = pipeline.transform(datos_cliente)

                if transformed_cliente.empty:
                    st.warning("El cliente seleccionado no cumple con los criterios de limpieza de datos y no puede ser procesado por el modelo.")
                else:
                    # Hacer la predicción
                    prediccion = model.predict(transformed_cliente)
                    resultado = prediccion[0]

                    # Mostrar el resultado de forma destacada
                    st.subheader(f"Resultado para el Cliente {cliente_id_seleccionado}")
                    st.metric("Clasificación Predicha por el Modelo", f"Clase {resultado}")

                    st.info(f"""
                    **Interpretación:** El modelo, basado en los datos históricos y de comportamiento del cliente,
                    lo ha clasificado en la **Clase {resultado}**.
                    """)

                    # Mostrar algunos datos clave del cliente para dar contexto
                    with st.expander("Ver datos clave del cliente utilizado para la predicción"):
                        st.dataframe(datos_cliente[['Socio', 'Producto', 'Saldo_total', 'Limite_credito', 'Utilizacion', 'Score_pago']])

            except Exception as e:
                st.error(f"No se pudo procesar al cliente. Error: {e}")