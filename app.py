#IMPORTAMOS LIBRERIAS
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import plotly_express as px
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html
from streamlit_folium import st_folium  
from folium.plugins import HeatMap
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')
#st.set_option('deprecation.showPyplotGlobalUse', False)

#CONFIGURACION DE LA PAGINA
st.set_page_config(
    page_title="AirBNB Oslo",
    page_icon="🇳🇴",
    layout="wide", # wide usa todo el ancho de la página mientras que centered centra el contenido en una columna
    initial_sidebar_state="expanded", #opciones: collapsed, expanded NO OBLIGATORIO
)

st.markdown(
    """
    <style>
    /* Fondo oscuro para toda la aplicación */
    .stApp {
        background-color: #1e1e2f;
        color: #dcdcdc;
    }

    /* Centrar y expandir contenido principal */
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Estilo para la barra lateral */
    .css-1d391kg, .css-qbe2hs {
        background-color: #29293d !important;
        color: #dcdcdc !important;
    }

    /* Botones y elementos interactivos */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: 1px solid #dcdcdc;
    }

    /* Campos de entrada */
    .stTextInput>div>input {
        background-color: #29293d;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#st.image("Oslo.png", use_column_width=True, caption="Vista de Oslo")


#FUNCION PARA CARGAR EL DATASET
df= pd.read_csv('listings.csv')

agg_data = df.groupby('neighbourhood').agg({
    'price': 'mean',
    'availability_365': lambda x: (x.sum() / len(x)) * 100  # Tasa de ocupación como porcentaje
}).reset_index()

# Renombrar columnas para facilitar el trabajo
agg_data.columns = ['neighbourhood', 'avg_price', 'occupancy_rate']

# Ordenar los vecindarios según el precio medio
agg_data = agg_data.sort_values('avg_price', ascending=False)

# Identificar los 5 vecindarios más caros, más baratos y el resto
top_5_expensive = agg_data.head(5)
top_5_cheap = agg_data.tail(5)

# Crear una columna para asignar los colores
def get_color(neighbourhood):
    if neighbourhood in top_5_expensive['neighbourhood'].values:
        return 'red'  # Color para los 5 más caros
    elif neighbourhood in top_5_cheap['neighbourhood'].values:
        return 'blue'  # Color para los 5 más baratos
    else:
        return 'green'  # Color para el resto

# Crear el mapa base
mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

# Añadir capa de mapa de calor
HeatMap(data=df[['latitude', 'longitude', 'price']].values, radius=10, max_zoom=13).add_to(mapa)

# Añadir marcadores con etiquetas para cada vecindario
for _, row in agg_data.iterrows():
    # Obtener las coordenadas aproximadas del vecindario (media de latitud y longitud)
    neigh_data = df[df['neighbourhood'] == row['neighbourhood']]
    lat = neigh_data['latitude'].mean()
    lon = neigh_data['longitude'].mean()

    # Obtener el color para el marcador
    color = get_color(row['neighbourhood'])

    # Crear el marcador en el mapa
    folium.CircleMarker(
        location=(lat, lon),
        radius=7,  # Tamaño del marcador
        color=color,
        fill=True,
        fill_opacity=0.7,
        tooltip=(f"<b>Vecindario:</b> {row['neighbourhood']}<br>"
                 f"<b>Precio Medio:</b> ${row['avg_price']:.2f}<br>"
                 f"<b>Tasa de Ocupación:</b> {row['occupancy_rate']:.2f}%")
    ).add_to(mapa)

#TITULO DE APLICACION
st.title('Análisis de Airbnb en la ciudad de Oslo')

#SIDEBAR
#st.sidebar.image('airbnb.png', width=200)
#st.sidebar.title('TITANIC')

#TABS
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Introduccion", "Evolucion del Precio", "Analisis Geográfico", 'Tasa de Ocupación', 'Numero de Reseñas','PowerBi','Conclusiones'])
with tab1:
    st.markdown("""
    **Oslo** es una ciudad que combina historia, cultura moderna y naturaleza. Con su impresionante fiordo y atracciones únicas, recibe más de **2 millones de visitantes al año**.

    Por su parte, **Airbnb** ha revolucionado la forma en que los viajeros experimentan Oslo, ofreciendo alojamientos variados y una conexión auténtica con la ciudad.
    """)

    # Lista de puntos destacados
    st.markdown("""
    ### ¿Qué encontrarás en este análisis?
    """)

    col1, col2, col3 = st.columns(3)

# Contenido de cada categoría
with col1:
    st.subheader("📈 Evolución de Precios")
    st.write("""
    - Análisis de la variación de precios a lo largo del tiempo.
    - Identificación de tendencias y eventos relevantes.
    """)
    st.progress(80)  # Ejemplo visual para reforzar la importancia de esta categoría.

with col2:
    st.subheader("📍 Análisis Geográfico")
    st.write("""
    - Mapa de calor para visualizar la distribución de precios y la tasa de ocupación.
    - Vecindarios más caros y más económicos.
    """) 

with col3:
    st.subheader("🛌 Tipología de Alojamiento")
    st.write("""
    - Relación entre precio y número de reseñas.
    - Comparativa entre Superhosts y hosts regulares.
    """)



with tab2:
    st.write('Evolución de la cantidad de Airbnb a lo largo de los años')
    evolution_per_year = df.groupby('year')['calculated_host_listings_count'].sum().reset_index()

    # Crear la figura
    fig = go.Figure()

    # Añadir el gráfico de barras
    fig.add_trace(go.Bar(
        x=evolution_per_year['year'],
        y=evolution_per_year['calculated_host_listings_count'],
        name='Cantidad total de listings',
        opacity=0.7
    ))

    # Añadir la línea de evolución
    fig.add_trace(go.Scatter(
        x=evolution_per_year['year'],
        y=evolution_per_year['calculated_host_listings_count'],
        mode='lines+markers',
        name='Evolución',
        line=dict(color='red', width=2),
        marker=dict(symbol='x', size=8)
    ))

    # Configurar título, ejes, fondo y eliminar grid
    fig.update_layout(
        xaxis_title='Año',
        yaxis_title='Cantidad de Airbnbs',
        legend=dict(x=0.1, y=0.9),
        plot_bgcolor="#1e1e2f",  # Fondo de la gráfica
        paper_bgcolor="#1e1e2f",  # Fondo fuera del área de la gráfica
        font=dict(color="white"),  # Color del texto
        xaxis=dict(showgrid=False),  # Quitar líneas del grid en eje X
        yaxis=dict(showgrid=False),  # Quitar líneas del grid en eje Y
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)




with tab3:
    st.write('Analisis geográfico')
    st_folium(mapa, width=1000)

    neighbourhood_prices = df.groupby('neighbourhood')['price'].mean().reset_index()

    # Ordenar por precio medio
    neighbourhood_prices = neighbourhood_prices.sort_values('price', ascending=False)

    # Crear el gráfico de barras con Plotly Express
    fig = px.bar(
        neighbourhood_prices,
        title='Precio medio por vecindario',
        x='neighbourhood',
        y='price',
        labels={'neighbourhood': 'Vecindario', 'price': 'Precio Medio ($)'},
        color='price',
        color_continuous_scale=[[1, "red"], [0, "white"]]
    )

    # Ajustar la rotación de los ticks del eje x
    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='#1e1e2f',  # Fondo del área del gráfico
        paper_bgcolor='#1e1e2f',  # Fondo general del gráfico
        font=dict(color='white'),  # Color del texto
        title_font=dict(size=20),  # Tamaño del título
        xaxis=dict(showgrid=False),  # Ocultar líneas del grid en eje X
        yaxis=dict(showgrid=False)   # Ocultar líneas del grid en eje Y
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    

with tab4:
    st.write('Tasa de Ocupación')

with tab5:
    st.write('Numero de Reseñas')

with tab6:
    st.write('Power Bi')

with tab7:
    st.write('Conclusiones')