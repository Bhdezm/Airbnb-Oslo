#IMPORTAMOS LIBRERIAS
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import plotly_express as px
import numpy as np
import streamlit.components.v1 as components
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
        background-color: #1e1e2f !important;  /* Fondo oscuro para toda la página */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Fondo y texto de la barra lateral */
    .css-1d391kg, .css-1y4p8pa, .css-qbe2hs, .stSidebar {
        background-color: #1e1e2f !important;  /* Fondo oscuro igual al resto */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Texto de títulos y subtítulos */
    .css-h3b2pw, h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;  /* Encabezados en blanco */
    }

    /* Fondo oscuro y texto blanco en selectores desplegables */
    .css-1l02zno, .stSelectbox, .stRadio, .stMultiselect {
        background-color: #29293d !important;  /* Fondo oscuro */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Bordes en los filtros desplegables */
    .css-1d391kg select {
        background-color: #29293d !important; /* Fondo oscuro */
        color: white !important; /* Texto blanco */
        border-color: #ffffff !important; /* Bordes blancos */
    }

    /* Fondo oscuro en los campos de texto */
    .stTextInput>div>input, .stTextArea>div>textarea {
        background-color: #29293d !important; /* Fondo oscuro */
        color: #ffffff !important;  /* Texto blanco */
    }

    /* Fondo y texto de los botones */
    .stButton>button {
        background-color: #ff4b4b !important; /* Fondo rojo */
        color: #ffffff !important; /* Texto blanco */
        border: 1px solid #ffffff !important; /* Borde blanco */
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
    st.subheader('Evolución de la cantidad de Airbnb a lo largo de los años')
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
        xaxis=dict(
        showgrid=False, 
        tickfont=dict(color='white')  # Color de las etiquetas en el eje X
    ),
    yaxis=dict(
        showgrid=False,  
    )
    )

    # Mostrar la gráfica en Streamlit
    st.plotly_chart(fig)

    st.subheader("Reflexión sobre la Evolución de Airbnbs")
    st.write("""
    - El año **2016** marcó el pico más alto de nuevos Airbnbs en Oslo, indicando un crecimiento exponencial en la popularidad de la plataforma.
    - A partir de **2020**, debido a la pandemia de COVID-19, se registró una drástica caída en la cantidad de Airbnbs, alcanzando su nivel más bajo desde 2014.
    - En los años posteriores, se observa una **recuperación gradual**, aunque la cantidad actual de listados sigue estando por debajo del máximo alcanzado en 2016.
    """)

    # Detalles adicionales o reflexiones finales
    st.info("""
    Este análisis refleja cómo eventos globales, como la pandemia, pueden impactar drásticamente el mercado de alquileres a corto plazo.
    Sin embargo, la tendencia de recuperación sugiere que la confianza en plataformas como Airbnb está retornando.
    """)

with tab3:
    st.subheader('Análisis geográfico')

    st_folium(mapa, width=1000)

    st.subheader('Precio Medio por vecindario')

    neighbourhood_prices = df.groupby('neighbourhood')['price'].mean().reset_index().sort_values('price', ascending=False)


    # Crear el gráfico de barras con Plotly Express
    fig = px.bar(
        neighbourhood_prices,
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
        font=dict(color='white'),  # Color del texto  # Tamaño del título
        xaxis=dict(showgrid=False),  # Ocultar líneas del grid en eje X
        yaxis=dict(showgrid=False)   # Ocultar líneas del grid en eje Y
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig, use_container_width=True)

    

with tab4:
    fig = px.scatter(filtered_data, x='price', y='occupancy_rate', 
                color='neighbourhood', 
                labels={'price': 'Precio', 'occupancy_rate': 'Tasa de Ocupación (%)'})

    fig.update_layout(
    plot_bgcolor='#1e1e2f',  # Fondo del gráfico
    paper_bgcolor='#1e1e2f',  # Fondo general
    font_color='white',  # Color del texto  # Estilo del título
    xaxis=dict(
        showgrid=False, 
        tickfont=dict(color='white')  # Color de las etiquetas en el eje X
    ),
    yaxis=dict(
        showgrid=True,  # Mostrar grid en eje Y
        gridcolor='gray',
        tickfont=dict(color='white')  # Color de las etiquetas en el eje Y
    ),
    legend=dict(
        title=dict(font=dict(color='white')),  # Color del título de la leyenda
        font=dict(color='white')  # Color del texto de la leyenda
    )  # Color de las líneas del grid (opcional)
)

# Mostrar el gráfico en Streamlit
    st.title("Análisis de Precio y Tasa de Ocupación")
    st.write("Este gráfico muestra la relación entre el precio y la tasa de ocupación, diferenciados por vecindarios.")
    st.plotly_chart(fig)

    fig = px.box(filtered_data, x='host_is_superhost', y='occupancy_rate', 
            labels={'superhost': 'Tipo de Host', 'occupancy_rate': 'Tasa de Ocupación (%)'})

    fig.update_layout(
    plot_bgcolor='#1e1e2f',  # Fondo del gráfico
    paper_bgcolor='#1e1e2f',  # Fondo general
    font_color='white',
    xaxis=dict(
        showgrid=False, 
        tickfont=dict(color='white')  # Color de las etiquetas en el eje X
    ),
    yaxis=dict(
        showgrid=True,  # Mostrar grid en eje Y
        gridcolor='gray',
        tickfont=dict(color='white')  # Color de las etiquetas en el eje Y
    ),
    legend=dict(
        title=dict(font=dict(color='white')),  # Color del título de la leyenda
        font=dict(color='white')  # Color del texto de la leyenda
    )  # Color del texto  # Estilo del título
)

# Mostrar el gráfico en Streamlit
    st.title("Análisis de Precio y Tasa de Ocupación")
    st.write("Este gráfico muestra la tasa de ocupación diferenciados por vecindarios.")
    st.plotly_chart(fig)

    grouped_data = filtered_data.groupby(['neighbourhood', 'host_is_superhost'], as_index=False).agg({'price': 'mean'})

    st.subheader('Precio medio por Superhost')
# Crear gráfico de barras apiladas
    fig = px.bar(
    grouped_data,
    x='neighbourhood',
    y='price',
    color='host_is_superhost', 
    labels={
        'neighbourhood': 'Vecindario',
        'price': 'Precio Medio ($)',
        'host_is_superhost': 'Tipo de Anfitrión'
    },
    color_discrete_map={
        't': 'blue', 
        'f': 'orange'  
    }
    )

# Ajustar la rotación de los ticks del eje x y diseño
    fig.update_layout(
    xaxis_tickangle=-45, 
    plot_bgcolor='#1e1e2f',  
    paper_bgcolor='#1e1e2f',
    font=dict(color='white'),  
    xaxis=dict(
        showgrid=False, 
        tickfont=dict(color='white')  # Color de las etiquetas en el eje X
    ),
    yaxis=dict(
        showgrid=True,  # Mostrar grid en eje Y
        gridcolor='gray',
        tickfont=dict(color='white')  # Color de las etiquetas en el eje Y
    ),
    legend=dict(
        title=dict(font=dict(color='white')),  # Color del título de la leyenda
        font=dict(color='white')  # Color del texto de la leyenda
    )
    )

# Mostrar en Streamlit
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Informe de Power BI - Airbnb Oslo")
    
    # Agregar iframe para incrustar el informe de Power BI
    st.markdown(
        """
        <iframe 
            title="Power BI Report" 
            width="100%" 
            height="800" 
            src="https://app.powerbi.com/view?r=eyJrIjoiYzM1NGQ3M2YtMDJhYi00ZTM2LTlhNzQtYjJkN2VlYzFjN2RmIiwidCI6IjhhZWJkZGI2LTM0MTgtNDNhMS1hMjU1LWI5NjQxODZlY2M2NCIsImMiOjl9" 
            frameborder="0" 
            allowFullScreen="true"></iframe>
        """,
        unsafe_allow_html=True
    )

with tab6:
    st.subheader("Precio y Vecindarios")
    st.write("""
        - Los vecindarios más caros se concentran al Oeste de Oslo, destacando **Nordstrand**, **Vestre Aker** y **Ullem**.
        - En comparación, vecindarios como **Stovner** o **Grorud** ofrecen precios más accesibles, atrayendo viajeros con menor presupuesto.
    """)

    # Impacto de los Superhosts
    st.subheader("Impacto de los Superhosts")
    st.write("""
        - Los superhosts tienden a cobrar un precio medio mayor en comparación con los anfitriones regulares.
        - Este aumento de precio puede justificarse por las **mejores reseñas** y la mayor confianza percibida por los usuarios a la hora elegir estancia.
    """)

    # Tasa de Ocupación
    st.subheader("Tasa de Ocupación y Disponibilidad")
    st.write("""
        - La tasa de ocupación más alta se observa en las zonas de **Grorud** y **Gamle Oslo**, situandose en la zona centro y a las afueras.
        - Estas Areas más baratas con una alta ocupación, sugieren oportunidades, ya que proporcionan una alta confianza en cuanto a la estancia.
    """)

    # Recomendaciones
    st.subheader("Recomendaciones")
    st.write("""
        - Para propietarios: Aumentar las reseñas, ofrecer servicios adicionales y convertirse en superhost puede aumentar el ingreso promedio y la tasa de ocupación.
        - Para viajeros: Explorar vecindarios más hacia las afueras como **Grorud** podría ofrecer una experiencia más auténtica a menor costo.
    """)
