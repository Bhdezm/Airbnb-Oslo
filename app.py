#IMPORTAMOS LIBRERIAS
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#st.set_option('deprecation.showPyplotGlobalUse', False)



#CONFIGURACION DE LA PAGINA
st.set_page_config(
    page_title="Análisis de Titanic",
    page_icon="🚢",
    layout="centered", # wide usa todo el ancho de la página mientras que centered centra el contenido en una columna
    initial_sidebar_state="expanded", #opciones: collapsed, expanded NO OBLIGATORIO
)


#FUNCION PARA CARGAR EL DATASET
df = sns.load_dataset('titanic')


#TITULO DE APLICACION
#st.title('Análisis de Titanic')

#IMAGEN
#st.image('img/titanic.jpg', width=200)

#TEXTOS
st.markdown('<h1 style="text-align: center; color: red;">Análisis de Titanic</h1>', unsafe_allow_html=True)
#st.write('Este es un análisis de los pasajeros del Titanic')


#SIDEBAR
#st.sidebar.image('img/titanic.jpg', width=200)
st.sidebar.title('TITANIC')


#COLUMNS
col1, col2, col3 = st.columns([1,2,1])
with col1:
    sns.countplot(data=df, x='survived')
    st.pyplot()

with col2:
    st.write('Este es un análisis de los pasajeros del Titanic')
with col3:
    sns.countplot(data=df, x='survived')
    st.pyplot()



#TABS
tab1, tab2, tab3 = st.tabs(["Análisis de Datos", "Machine Learning", "Predicción"])
with tab1:
    st.title('Análisis de Datos')
    st.write(df.head())


with tab2:
    sns.countplot(data=df, x='survived')
    st.pyplot()
    st.write('Este es un análisis de los pasajeros del Titanic, representa la cantidad de personas que sobrevivieron y las que no')


with tab3:
    sns.histplot(data=df, x='age', hue='survived', kde=True)
    st.pyplot()