import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import time

st.set_page_config(page_title="Iris Flower Predictor", layout="wide")

st.title("Iris Flower Species Prediction and Exploration")
st.write("Predict the species of an Iris flower based on its measurements, or explore the dataset to understand the distribution and relationships of its features.")

data, iris_data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names), load_iris()
data["species_name"] = [iris_data.target_names[i] for i in iris_data.target]

model_file = "iris_model_rf.joblib"
try:
    model = joblib.load(model_file)
except FileNotFoundError:
    X_train, X_test, y_train, y_test = train_test_split(
        data[iris_data.feature_names], iris_data.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    joblib.dump(model, model_file)

st.sidebar.header("App Sections")
section = st.sidebar.radio("Select Section", ["Flower Prediction", "Explore Data"])

if section == "Flower Prediction":
    st.subheader("Flower Prediction")
    st.write("Enter flower measurements below to predict the species.")
    sliders = {f: st.slider(f, float(data[f].min()), float(data[f].max()), float(data[f].mean()))
               for f in iris_data.feature_names}

    if st.button("Predict Species"):
        time.sleep(1)
        pred_class = model.predict(np.array([list(sliders.values())]))[0]
        st.success(f"Predicted Species: {iris_data.target_names[pred_class]}")
        st.balloons()

elif section == "Explore Data":
    st.subheader("Iris Dataset Overview")
    st.write("This section lets you explore the dataset visually and statistically.")
    st.snow()

    if st.checkbox("Show Dataset Table"):
        st.dataframe(data)

    viz_type = st.selectbox("Select Visualization", ["Histogram", "Scatter Plot"])

    if viz_type == "Histogram":
        feature = st.selectbox("Select Feature for Histogram", iris_data.feature_names)
        st.info(f"This shows the distribution of {feature}.")
        fig, ax = plt.subplots(figsize=(3,2))
        sns.histplot(data=data, x=feature, bins=20, kde=True, ax=ax)
        ax.set_title(f"Histogram of {feature}", fontsize=10)
        ax.set_xlabel(feature, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        st.pyplot(fig)

    elif viz_type == "Scatter Plot":
        x_feature = st.selectbox("X-axis Feature", iris_data.feature_names, index=0)
        y_feature = st.selectbox("Y-axis Feature", iris_data.feature_names, index=1)
        st.info(f"This scatter plot shows {y_feature} vs {x_feature}.")
        fig, ax = plt.subplots(figsize=(3,2))
        sns.scatterplot(data=data, x=x_feature, y=y_feature, hue="species_name", palette="Set2", ax=ax, s=25)
        ax.set_title(f"{y_feature} vs {x_feature}", fontsize=10)
        ax.set_xlabel(x_feature, fontsize=8)
        ax.set_ylabel(y_feature, fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(fontsize=6)
        st.pyplot(fig)

st.markdown("""
About
- This app is built using Streamlit
- Model: RandomForestClassifier trained on the Iris dataset
- Dataset: Iris (Scikit-learn)
""")