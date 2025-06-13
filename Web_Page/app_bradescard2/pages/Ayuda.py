# pages/3_❓_Ayuda.py

import streamlit as st

st.set_page_config(page_title="Ayuda❓", layout="wide")

st.title("Guía de Usuario y Ayuda")

st.info(
    "Esta guía te ayudará a entender cómo funciona cada parte de la aplicación."
)

st.header("Página de Inicio")
st.markdown(
    """
    La página de **Inicio** es tu punto de partida. Aquí puedes:
    - **Elegir tu fuente de datos**:
        - **Usar datos de demostración**: Carga un conjunto de datos de ejemplo para que puedas probar la aplicación sin necesidad de tener tus propios datos.
        - **Subir mi propio archivo**: Te permite cargar un archivo en formato `.csv`.
    - **Requisitos del archivo**:
        - Debe ser un archivo CSV.
        - Es **crucial** que contenga una columna llamada `Variable_objetivo`. Esta es la variable que el modelo intentará predecir.
    - **Vista previa**: Una vez cargado, verás las primeras filas de tu tabla para confirmar que se ha leído correctamente.
    """
)

st.header("Página de Análisis y Modelo")
st.markdown(
    """
    Esta es la sección donde ocurre la magia.
    1.  **Iniciar Análisis**: Haz clic en el botón **"Iniciar Análisis y Entrenamiento del Modelo"**.
    2.  **Procesamiento**: La aplicación ejecutará automáticamente el pipeline de ciencia de datos:
        - **Limpieza de Datos**: Se corrigen formatos, se manejan valores faltantes y se transforman las fechas.
        - **Transformación**: Se convierten las variables categóricas a un formato numérico que el modelo pueda entender (`One-Hot Encoding`).
        - **Escalado**: Se normalizan los datos para mejorar el rendimiento del modelo.
        - **Reducción de Dimensionalidad (PCA)**: Se simplifica la complejidad de los datos, conservando la información más importante.
    3.  **Entrenamiento**: Se entrena un modelo de `RandomForestClassifier` para aprender de tus datos.
    4.  **Resultados**:
        - **Precisión del Modelo**: Un porcentaje que te dice qué tan bueno es el modelo haciendo predicciones.
        - **Reporte de Clasificación**: Métricas más detalladas (precisión, recall, f1-score) que te dan una idea del rendimiento del modelo para cada clase de la variable objetivo.
        - **Factores Clave**: Un gráfico que te muestra las 15 variables más importantes que el modelo utiliza para tomar sus decisiones. ¡Ideal para obtener insights de negocio!
    """
)

st.header("Consejos Adicionales")
st.markdown(
    """
    - **Paciencia**: El proceso de entrenamiento puede tardar varios minutos, especialmente con archivos grandes.
    - **Errores**: Si encuentras un error, lo más común es un problema con el formato del archivo o la ausencia de la `Variable_objetivo`. Revisa tu archivo y vuelve a intentarlo.
    """
)