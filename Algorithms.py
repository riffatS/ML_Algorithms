
# from calendar import c
import sklearn
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use( 'TkAgg')

st.write(
    """
    # Explore Different Classifiers With Different Dataset
    """
)
st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Database")
dataset_name=st.sidebar.selectbox("Select dataset",("Iris","Breast Cancer", "Diabetes","Wine"))
classifier_name=st.sidebar.selectbox("Select classifier",("KNN","SVM", "Random Forest"))

def get_database(dataset_name):
    if dataset_name =="Iris":
        data=datasets.load_iris()
        
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif dataset_name=="Diabetes":
        data=datasets.load_diabetes()
    # elif dataset_name=="Housing Prices":
    #     data=datasets.load_boston()

    else:
        data=datasets.load_wine()
    
    X=data.data
    Y=data.target
    # st.write(data)
    return (X,Y,data)

def data_info(data):
    st.write(data.DESCR)
    n_samples, n_features = data.data.shape
    st.write('Number of samples:', n_samples)
    st.write('Number of features:', n_features)
    # the sepal length, sepal width, petal length and petal width of the first sample (first flower)
    print(data.data[0])









def add_parameter(classifier_name):
    param=dict()
    if classifier_name == "KNN":
        K=st.sidebar.slider("K",1,15)
        param["K"]=K
    elif classifier_name == "SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        param["C"]=C
    elif classifier_name == "Random Forest" :
    
        max_depth=st.sidebar.slider("max_depth", 2,12)
        n_estimator=st.sidebar.slider("n_estimator", 1,100)
        param["max_depth"]=max_depth
        param["n_estimator"]=n_estimator
    return param
    
def get_classifier(classifier_name,param):
    
    if classifier_name == "KNN":
        classifier=KNeighborsClassifier(n_neighbors=param["K"])
    elif classifier_name == "SVM":
        classifier=SVC(C=param["C"])
    else:
        classifier=RandomForestClassifier(n_estimators=param["n_estimator"],max_depth=param["max_depth"],random_state=1234)
   
    return classifier


def classify_data(X,Y,claf):
        
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1234)
    claf.fit(X_train,Y_train)
    y_pred=claf.predict(X_test)
    accuracy=accuracy_score(Y_test,y_pred)
    return accuracy


#plot
def plot(X,Y):

    pca=PCA(2)
    X_reduced = pca.fit_transform(X)
    x1=X_reduced[:,0]
    x2=X_reduced[:,:1]
    fig = plt.subplots()
    plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
    plt.colorbar()
    # plt.show()
    st.pyplot()
    plt.close()


X,Y,data=get_database(dataset_name)
param=add_parameter(classifier_name)
claf=get_classifier(classifier_name,param)
accuracy=classify_data(X,Y,claf)
st.write(X)
st.title("Graph")
plot(X,Y)


st.title("Classifier details")
st.write(f"Classifier Name ={classifier_name}")
st.write(f"Classifier Accuracy ={accuracy}")

data_info(data)


# classifier 




