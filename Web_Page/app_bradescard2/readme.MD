High-Level Workflow
The application's workflow is organized into distinct pages, accessible via a sidebar for navigation. A user starts at the "Inicio" (Home) page to load data, which is then stored in a session state, making it available across all other analytical pages. The user can then navigate to pages for a general portfolio overview, risk analysis, and help.

Detailed Page Descriptions
Inicio.py (Home Page)
This script serves as the application's entry point and is dedicated to data loading.

Data Source Options: It offers two methods for data input: using a pre-loaded demonstration file named COLL_TEC_CONSOLIDADO.txt or uploading a custom file.
File Upload: The uploader accepts .csv and .txt files. The code attempts to read these files using a comma delimiter and latin-1 encoding to handle special characters.
Large File Handling: It explicitly informs the user that to upload files larger than the 200MB default limit (up to 1GB), the Streamlit server must be launched with a specific command: streamlit run Inicio.py --server.maxUploadSize=1024.
Data Persistence and Preview: Upon a successful load, the DataFrame is stored in Streamlit's session_state under the key 'df', allowing other pages to access it without reloading. A preview of the first few rows of the dataset is then displayed on the page.
Análisis_de_Cartera.py (Portfolio Analysis Page)
This page functions as a general dashboard for visualizing the client portfolio.

Key Metrics: It calculates and displays three key metrics at the top: the total number of clients, the total portfolio balance (Saldo_total), and the average credit utilization (Utilizacion) formatted as a percentage.
Visualizations:
A pie chart visualizes the proportion of clients associated with each "Socio" (Partner).
A histogram shows the frequency distribution of the Saldo_total across the entire portfolio, with 50 bins for granularity.
Individual Client Search: It features a dropdown menu (selectbox) where a user can select a client by a unique ID_Cliente. If an ID column doesn't exist, it is created. Selecting a client displays their detailed profile, neatly organized into sections like "Información Principal," "Estado Financiero Actual," and "Comportamiento de Pago".
Analisis_de_Riesgo.py (Risk Analysis Page)
This is the most technically advanced module, performing client segmentation using two different methods.

Rule-Based Risk Segmentation:
It defines a function to assign a risk level ('Bajo Riesgo', 'Riesgo Medio', 'Alto Riesgo') based on the value of Score_pago (Payment Score).
The results are shown in a bar chart colored by risk level and a summary table that aggregates average balance, average utilization, and client count for each risk segment.
Machine Learning Segmentation (Clustering):
This analysis is triggered by a button click to avoid running it automatically.
Preprocessing: It prepares the data by selecting numerical features, filling null values with 0, and dropping columns not relevant for behavior analysis like Variable_objetivo and ID_Cliente.
Scaling: It uses StandardScaler to normalize the numerical data, which is a crucial step for distance-based algorithms like K-Means.
Clustering: It applies the KMeans algorithm to group clients into three distinct clusters (n_clusters=3). The resulting cluster labels are added to the DataFrame and saved in the session state as df_clustered.
Visualization: To visualize the high-dimensional clusters in 2D, it uses PCA (Principal Component Analysis) to reduce the data to two principal components, which are then plotted on a scatter plot colored by cluster.
Interpretation: A summary table is provided to help interpret the characteristics of each cluster, showing the average balance, utilization, payment score, and number of clients for each group.
Modelo_Predictivo.py (Predictive Model Page)

This page is explicitly marked as a placeholder for a future feature.
It is described as a "Módulo en Desarrollo" (Module in Development).
The stated purpose is to eventually house a predictive model (like Random Forest) to predict the Variable_objetivo and identify key predictive factors.
Ayuda.py (Help Page)

This script acts as a user manual for the application.
It explains the functionality of each page, providing clarity on the data science processes being executed.
Crucially, it highlights the requirement for any user-uploaded file to contain a Variable_objetivo column, which is essential for the predictive modeling parts of the application.