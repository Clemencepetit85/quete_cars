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

st.title('Hello Wilders, welcome to my application!')

st.write("I enjoy to discover streamlit possibilities")

link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/weather2019.csv"
df_weather = pd.read_csv(link)
st.write(df_weather)

st.line_chart(df_weather['MAX_TEMPERATURE_C'])


name = st.text_input("Please give me your name :")
name_length = len(name)
st.write("Your name has ",name_length,"characters")
