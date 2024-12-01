#IMPORTAMOS LIBRERIAS
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from streamlit_folium import st_folium  
from folium.plugins import HeatMap
import warnings

warnings.filterwarnings('ignore')
#st.set_option('deprecation.showPyplotGlobalUse', False)

#CONFIGURACION DE LA PAGINA
st.set_page_config(
    page_title="AirBNB Oslo",
    page_icon="🇳🇴",
    layout="centered", # wide usa todo el ancho de la página mientras que centered centra el contenido en una columna
    initial_sidebar_state="expanded", #opciones: collapsed, expanded NO OBLIGATORIO
)

st.markdown("""
    <style>
    body {
        background-color: black;
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)


#FUNCION PARA CARGAR EL DATASET
listings= pd.read_csv('gz_listings.csv')
listings = listings[['latitude', 'longitude', 'price', 'neighbourhood']]
df=pd.read_csv('listings.csv')
columnas_numericas = listings.select_dtypes(include=['number'])

# Inicializar el KNNImputer con k = 5 (puedes cambiar el valor según lo analizado)
imputer = KNNImputer(n_neighbors=5)

# Imputar valores nulos en las columnas numéricas
df_numerico_imputado = imputer.fit_transform(columnas_numericas)

# Crear un nuevo DataFrame con los valores imputados y mismas columnas
df_imputado = pd.DataFrame(df_numerico_imputado, columns=columnas_numericas.columns)

# Reemplazar las columnas numéricas originales en el DataFrame original con las imputadas
listings[columnas_numericas.columns] = df_imputado

mapa = folium.Map(location=[listings['latitude'].mean(), listings['longitude'].mean()], zoom_start=10)

# Añadir capa de mapa de calor (sin etiquetas, solo el heatmap)
HeatMap(data=listings[['latitude', 'longitude', 'price']].values, radius=10, max_zoom=13).add_to(mapa)

# Añadir marcadores con etiquetas para cada punto con su 'neighbourhood'
for idx, row in listings.iterrows():
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=3,  # Radio pequeño para que no se solape con el heatmap
        color='#87CEFA',
        fill=True,
        fill_opacity=0.6,
        tooltip=f"<b>Barrio:</b> {row['neighbourhood']}<br><b>Precio:</b> ${row['price']:.2f}"
        #popup=folium.Popup(f"<b>Barrio:</b> {row['neighbourhood']}<br><b>Precio:</b> ${row['price']:.2f}", max_width=300)
    ).add_to(mapa)

#TITULO DE APLICACION
#st.title('Análisis de Titanic')

#IMAGEN
#st.image('img/titanic.jpg', width=200)

#TEXTOS
st.markdown('<h1 style="text-align: center; color: red;">Análisis de Airbnb en Oslo</h1>', unsafe_allow_html=True)
#st.write('Este es un análisis de los pasajeros del Titanic')


#SIDEBAR
#st.sidebar.image('img/titanic.jpg', width=200)
#st.sidebar.title('TITANIC')

#TABS
tab1, tab2, tab3 = st.tabs(["Mapa de Calor", "Evolucion de Airbnb", "Predicción"])
with tab1:
    st.write('Mapa de calor de los Airbnb en la ciudad de Oslo')
    st_folium(mapa, width=1000)


with tab2:
    st.write('Evolución de la cantidad de Airbnb a lo largo de los años')
    df['host_since'] = pd.to_datetime(df['host_since'])

# Extraer el año de 'host_since'
    df['year'] = df['host_since'].dt.year

# Agrupar por año y calcular la suma de 'calculated_host_listings_count'
    evolution_per_year = df.groupby('year')['calculated_host_listings_count'].sum()

# Crear la figura y el gráfico de barras
    fig, ax1 = plt.subplots(figsize=(12, 6))

# Gráfico de barras
    ax1.bar(evolution_per_year.index, evolution_per_year.values, color='skyblue', alpha=0.7, label='Cantidad total de listings')
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Cantidad de Airbnbs')
    ax1.tick_params(axis='y')

# Crear un segundo eje para la línea
    ax2 = ax1.twinx()
    ax2.plot(evolution_per_year.index, evolution_per_year.values, color='r', marker='x', label='Evolución')
    ax2.set_ylabel('Evolución de Airbnbs')
    ax2.tick_params(axis='y')

# Título y leyenda
    #plt.title('Evolución de la cantidad de Airbnbs a lo largo de los años')
    fig.tight_layout()
    #fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    st.pyplot(fig)


with tab3:
    st.title('Relación entre numero de reseñas y precio')
    plt.figure(figsize=(50,25))
    sns.relplot(data = df, x = 'number_of_reviews', y = 'price', hue = 'room_type')
    st.pyplot()