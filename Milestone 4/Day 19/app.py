import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import seaborn as sn

st.set_page_config(page_title="Iris Flower Prediction App", layout="wide")
st.title("🌸 Iris Flower Classification Web App")
st.markdown("""
This interactive app allows you to:
- Input **iris flower features** and get predictions from a trained Random Forest model.  
- Explore the **Iris dataset** using visualizations in the sidebar.  
""")

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris

data, iris_data = load_data()

model_filename = "iris_rf_model.joblib"

def train_and_save_model():
    X_train, X_test, y_train, y_test = train_test_split(
        data[iris_data.feature_names], data['target'],
        test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    return model

try:
    model = joblib.load(model_filename)
except FileNotFoundError:
    model = train_and_save_model()

mode = st.sidebar.radio("Choose Mode:", ["Prediction", "Data Exploration"])

if mode == "Prediction":
    st.header("🔮 Make a Prediction")
    st.markdown("Adjust the sliders below to set flower measurements:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider(
            'Sepal Length (cm)',
            float(data['sepal length (cm)'].min()),
            float(data['sepal length (cm)'].max()),
            float(data['sepal length (cm)'].mean())
        )
        sepal_width = st.slider(
            'Sepal Width (cm)',
            float(data['sepal width (cm)'].min()),
            float(data['sepal width (cm)'].max()),
            float(data['sepal width (cm)'].mean())
        )
    with col2:
        petal_length = st.slider(
            'Petal Length (cm)',
            float(data['petal length (cm)'].min()),
            float(data['petal length (cm)'].max()),
            float(data['petal length (cm)'].mean())
        )
        petal_width = st.slider(
            'Petal Width (cm)',
            float(data['petal width (cm)'].min()),
            float(data['petal width (cm)'].max()),
            float(data['petal width (cm)'].mean())
        )

    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(input_features)[0]
        prediction_proba = model.predict_proba(input_features)[0]

        st.success(f"🌼 Predicted Species: **{iris_data.target_names[prediction]}**")

        proba_df = pd.DataFrame({
            'Species': iris_data.target_names,
            'Probability': prediction_proba
        })

        st.subheader("Prediction Probabilities")
        st.table(proba_df)
        st.bar_chart(proba_df.set_index("Species"))

elif mode == "Data Exploration":
    st.header("📊 Data Exploration")
    st.markdown("Explore the Iris dataset with simple visualizations using the sidebar.")

    explore_option = st.sidebar.selectbox(
        "Select Visualization",
        ["Histogram", "Scatter Plot", "Pairplot", "Correlation Heatmap"]
    )

    if explore_option == "Histogram":
        feature = st.sidebar.selectbox("Select Feature for Histogram", iris_data.feature_names)
        fig, ax = plt.subplots()
        sns.histplot(data[feature], bins=20, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram of {feature}")
        st.pyplot(fig)

    elif explore_option == "Scatter Plot":
        x_feature = st.sidebar.selectbox("Select X-axis Feature", iris_data.feature_names, index=0)
        y_feature = st.sidebar.selectbox("Select Y-axis Feature", iris_data.feature_names, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=x_feature, y=y_feature, hue='species', palette='Set1', ax=ax, s=100)
        ax.set_title(f"Scatter Plot of {y_feature} vs {x_feature}")
        st.pyplot(fig)

    elif explore_option == "Pairplot":
        st.info("This might take a few seconds to generate.")
        fig = sns.pairplot(data, hue="species", diag_kind="kde", palette="Set2")
        st.pyplot(fig)

    elif explore_option == "Correlation Heatmap":
        fig, ax = plt.subplots()
        sns.heatmap(data[iris_data.feature_names].corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

st.markdown("---")
st.markdown("🚀 Developed with **Streamlit & scikit-learn**")