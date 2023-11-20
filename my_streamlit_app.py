import streamlit as st
#cleaning and aggregation
from collections import Counter
import numpy as np
import math as m
#read data
from sklearn.datasets import load_breast_cancer
#dataviz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#split train test
from sklearn.model_selection import train_test_split
#scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#ml model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#ml score
from sklearn import metrics
#data viz arbre decision
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from sklearn.metrics  import silhouette_score
from sklearn.cluster import KMeans


##################################################################

# élargir le contenu
st.set_page_config(layout="wide")

###############################################################

st.title('Hello Wilders, welcome to my application!')

##################################################################

st.markdown("""
<div style="width: 90%;">
Vous trouverez ci-dessous une analyse du dataframe "<a href="https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv" target="_blank style="color: black;">cars</a>" qui présente les caractéristiques de modèles de voiture européennes, japonaises et américaines.<br> 
Vous pouvez sélectionner une région pour afficher les différents graphiques de corrélation liées aux voitures de cette dernière. <br> 
Sous les graphiques, des commentaires vous aideront à les interpréter.<br> 
Bonne lecture !
<br>
<hr style="border: 1px solid black;">
<br>
</div>
""", unsafe_allow_html=True)

#https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv

##################################################################

#import du dataframe cars
link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df_cars = pd.read_csv(link)

####################################################################

# Interface utilisateur avec Streamlit

#création d'un outil pour choisir la région

# Filtrer les données par région
regions = df_cars['continent'].unique()

selected_region = st.selectbox('Sélectionnez une région', regions)

filtered_data = df_cars[df_cars['continent'] == selected_region]

####################################################################

# Création d'une heatmap (on supprime donc d'abord la colonne continent)

df_heatmap = filtered_data.drop("continent", axis = 1)

plt.figure(figsize=(10, 8))
sns.heatmap(df_heatmap.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap de corrélation entre les variables')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
#st.pyplot(plt.gcf())

#################################################################


# Création des 4 graphiques avec une taille ajustée
fig1, ax1 = plt.subplots(figsize=(12, 9))
sns.scatterplot(x='cubicinches', y='weightlbs', data=filtered_data, ax=ax1)
ax1.set_xlabel('Volume du moteur')
ax1.set_ylabel('Poids de la voiture')
ax1.set_title(f'Corrélation : Poids de la voiture en fonction du volume du moteur pour les voitures de la région{selected_region}')

fig2, ax2 = plt.subplots(figsize=(12, 9))
sns.scatterplot(x='hp', y='weightlbs', data=filtered_data, ax=ax2)
ax2.set_xlabel('Nombre de chevaux')
ax2.set_ylabel('Poids de la voiture')
ax2.set_title(f'Corrélation : Poids de la voiture en fonction du nombre de chevaux pour les voitures de la région{selected_region}')

fig3, ax3 = plt.subplots(figsize=(12, 9))
sns.scatterplot(x='cubicinches', y='hp', data=filtered_data, ax=ax3)
ax3.set_xlabel('Volume du moteur')
ax3.set_ylabel('Nombre de chevaux')
ax3.set_title(f'Corrélation : Volume du moteur en fonction du nombre de chevaux pour les voitures de la région{selected_region}')

fig4, ax4 = plt.subplots(figsize=(12, 9))
sns.scatterplot(x='mpg', y='cylinders', data=filtered_data, ax=ax4)
ax4.set_xlabel('Miles per gallon')
ax4.set_ylabel('Nombre de cylindres')
ax4.set_title(f'Corrélation : Miles parcourues avec un gallon de carburant en fonction du nombre de cylindres pour les voitures de la région{selected_region}')

# Affichage des graphiques dans deux colonnes de deux lignes avec une largeur étendue
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1)
    st.pyplot(fig2)

with col2:
    st.pyplot(fig3)
    st.pyplot(fig4)


##################################################################

#affichage des commentaires par région sélectionnée 
# Créer une colonne pour afficher les commentaires
col1, col2, col3 = st.columns(3)

# Affichage des commentaires dédiés dans la deuxième colonne
if selected_region == " Europe.":

        st.markdown("""
        <div style="width: 90%; text-align: left;">
        <br>
        <hr style="border: 1px solid black;">
        Commentaires pour la région Europe :<br>
        Généralement, plus le volume du moteur de la voiture est important, plus le poids de la voiture augmente.<br>
        Les voitures ayant un nombre de chevaux plus élevé ont également tendance à peser plus lourd.<br>
        Ceci s'explique par le fait que le volume du moteur augmente avec le nombre de chevaux pour la plupart des voitures analysées.<br>
        Enfin, contrairement aux USA, on ne remarque pas de lien entre le fait de pouvoir faire davantage de kilomètres avec un même plein d'essence si le nombre de cylindres augmente.
        <br>
        <hr style="border: 1px solid black;">
        <br>
        </div>
        """, unsafe_allow_html=True)

elif selected_region == " US.":

        st.markdown("""
        <div style="width: 90%; text-align: left;">
        <br>
        <hr style="border: 1px solid black;">
        Commentaires pour la région USA :<br>
        Généralement, plus le volume du moteur de la voiture est important, plus le poids de la voiture augmente.<br>
        Les voitures ayant un nombre de chevaux plus élevé ont également tendance à peser plus lourd.<br>
        Ceci s'explique par le fait que le volume du moteur augmente avec le nombre de chevaux pour la plupart des voitures analysées.<br>
        Enfin, on remarque aussi que les voitures ayant un nombre plus élevé de cylindres peuvent faire davantage de kilomètres avec un même plein d'essence.
        <br>
        <hr style="border: 1px solid black;">
        <br>
        </div>
        """, unsafe_allow_html=True)

elif selected_region == " Japan.":

        st.markdown("""
        <div style="width: 90%; text-align: left;">
        <br>
        <hr style="border: 1px solid black;">
        Commentaires pour la région Japan :<br>
        Généralement, plus le volume du moteur de la voiture est important, plus le poids de la voiture augmente.<br>
        Les voitures ayant un nombre de chevaux plus élevé ont également tendance à peser plus lourd.<br>
        Ceci s'explique par le fait que le volume du moteur augmente avec le nombre de chevaux pour la plupart des voitures analysées.<br>
        Enfin, contrairement aux USA, on ne remarque pas de lien entre le fait de pouvoir faire davantage de kilomètres avec un même plein d'essence si le nombre de cylindres augmente.
        <br>
        <hr style="border: 1px solid black;">
        <br>
        </div>
        """, unsafe_allow_html=True)

else:
    with col2:
        st.markdown("""
        <div style="width: 90%; text-align: center;">
        <br>
        <hr style="border: 1px solid black;">
        Bonne journée !
        <br>
        <hr style="border: 1px solid black;">
        <br>
        </div>
        """, unsafe_allow_html=True)