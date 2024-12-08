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
        background-color: #1e1e2f !important;  /* Color de fondo oscuro */
        color: #ffffff !important;  /* Color de texto blanco */
    }

    /* Centrar y expandir contenido principal */
    .main {
        max-width: 1500px;
        margin: 0 auto;
        padding: 20px;
    }

    /* Barra lateral: Fondo oscuro y texto blanco */
    .stSidebar {
        background-color: #29293d !important; /* Fondo oscuro en la barra lateral */
        color: #ffffff !important; /* Texto blanco en la barra lateral */
    }

    /* Títulos y subtítulos en blanco */
    .stHeader, .stSubheader, .stTitle {
        color: #ffffff !important;  /* Títulos en blanco */
    }

    /* Títulos, etiquetas y textos en blanco */
    .stText, .stMarkdown, .stLabel, .stSelectbox, .stRadio, .stMultiselect {
        color: #ffffff !important;  /* Texto blanco en todos los elementos */
    }

    /* Fondo oscuro para los botones */
    .stButton>button {
        background-color: #ff4b4b !important; /* Color de fondo del botón */
        color: white !important; /* Texto blanco */
        border: 1px solid #dcdcdc !important;  /* Borde blanco */
    }

    /* Campos de entrada (textos) con fondo oscuro y texto blanco */
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #29293d !important; /* Fondo oscuro en los campos de texto */
        color: white !important;  /* Texto blanco en los campos de texto */
    }

    /* Seleccionadores desplegables con fondo oscuro y texto blanco */
    .stSelectbox, .stRadio, .stMultiselect {
        background-color: #29293d !important;  /* Fondo oscuro en los selectores */
        color: white !important;  /* Texto blanco */
    }

    /* Barra lateral: Controles y texto dentro de la barra lateral */
    .stSidebar .css-1l02zno {
        background-color: #29293d !important; /* Fondo oscuro en la parte superior de la barra lateral */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Color de borde en las gráficas */
    .js-plotly-plot .plotly {
        background-color: #29293d !important;  /* Fondo oscuro en los gráficos */
        color: white !important;  /* Texto blanco en los gráficos */
    }
    </style>
    """,
    unsafe_allow_html=True
)



#FUNCION PARA CARGAR EL DATASET
df= pd.read_csv('listings.csv')

st.sidebar.image('airbnb.png')

st.sidebar.title("Filtros")

# Filtro de vecindarios (menú desplegable)
neighbourhoods = df['neighbourhood'].unique()
selected_neighbourhood = st.sidebar.selectbox(
    "Selecciona un vecindario:", 
    options=["Todos"] + list(neighbourhoods), 
    index=0
)

# Filtro de Superhost (menú desplegable)
superhost_options = df['host_is_superhost'].unique()
selected_superhost = st.sidebar.selectbox(
    "Filtrar por Superhost:", 
    options=["Todos"] + list(superhost_options), 
    index=0
)

# Filtro de año (menú desplegable)
if 'year' in df.columns:
    years = sorted(df['year'].unique())
    selected_year = st.sidebar.selectbox(
        "Selecciona el año:", 
        options=["Todos"] + years,
        index=0
    )
else:
    selected_year = "Todos"

# Aplicar filtros
filtered_data = df

if selected_neighbourhood != "Todos":
    filtered_data = filtered_data[filtered_data['neighbourhood'] == selected_neighbourhood]

if selected_superhost != "Todos":
    filtered_data = filtered_data[filtered_data['host_is_superhost'] == selected_superhost]

if selected_year != "Todos":
    filtered_data = filtered_data[filtered_data['year'] == selected_year]


#st.image("Oslo.png", use_column_width=True, caption="Vista de Oslo")


agg_data = df.groupby('neighbourhood').agg({
    'price': 'mean',
    'occupancy_rate': 'mean'  # Tasa de ocupación como porcentaje
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Introducción", "Oferta Airbnb", "Analisis Geográfico", 'Tasa de Ocupación','PowerBi','Conclusiones'])
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

    neighbourhood_prices = df.groupby('neighbourhood')['price'].mean().reset_index().sort_values('price', ascending=False)


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
    st.write('Tasa de ocupación')
    fig = px.scatter(filtered_data, x='price', y='occupancy_rate', 
                color='neighbourhood', 
                labels={'price': 'Precio', 'occupancy_rate': 'Tasa de Ocupación (%)'})

    fig.update_layout(
    plot_bgcolor='#1e1e2f',  # Fondo del gráfico
    paper_bgcolor='#1e1e2f',  # Fondo general
    font_color='white',  # Color del texto  # Estilo del título
    xaxis=dict(gridcolor='#444444'),  # Color de las líneas del grid (opcional)
    yaxis=dict(gridcolor='#444444')  # Color de las líneas del grid (opcional)
)

# Mostrar el gráfico en Streamlit
    st.title("Análisis de Precio y Tasa de Ocupación")
    st.write("Este gráfico muestra la relación entre el precio y la tasa de ocupación, diferenciados por vecindarios.")
    st.plotly_chart(fig)

    st.write('Tasa de ocupación por diferenciado en superhost y regularhost')
    fig = px.box(filtered_data, x='host_is_superhost', y='occupancy_rate', 
            title='Tasa de Ocupación: Superhost vs Host Regular', 
            labels={'superhost': 'Tipo de Host', 'occupancy_rate': 'Tasa de Ocupación (%)'})

    fig.update_layout(
    plot_bgcolor='#1e1e2f',  # Fondo del gráfico
    paper_bgcolor='#1e1e2f',  # Fondo general
    font_color='white',
    xaxis=dict(gridcolor='#444444'),  # Color de las líneas del grid (opcional)
    yaxis=dict(gridcolor='#444444')  # Color del texto  # Estilo del título
)

# Mostrar el gráfico en Streamlit
    st.title("Análisis de Precio y Tasa de Ocupación")
    st.write("Este gráfico muestra la relación entre el precio y la tasa de ocupación, diferenciados por vecindarios.")
    st.plotly_chart(fig)

with tab5:
    st.write('Power Bi')

with tab6:
    st.write('Conclusiones')