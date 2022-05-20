import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np 
st.write(
    """
    # Explore Different Classifiers With Different Dataset
    """
)

st.title("hey")
dataset_name=st.sidebar.selectbox("Select dataset",("Iris","Breast Cancer", "Diabetes","Housing Prices","Wine"))
classifier_name=st.sidebar.selectbox("Select classifier",("KNN","SVM", "Random Forest","Naive Bayes"))

def get_database(dataset_name):
    if dataset_name =="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif dataset_name=="Diabetes":
        data=datasets.load_diabetes()
    elif dataset_name=="Housing Prices":
        data=datasets.load_boston()

    else:
        data=datasets.load_wine()
    X=data.data
    Y=data.target
    return (X,Y)

X,Y=get_database(dataset_name)
st.write(X)