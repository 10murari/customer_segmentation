# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation using KMeans & Hierarchical Clustering")

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Mall_Customers.csv")

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})

st.subheader("Data Preview")
# st.dataframe(df.head())
st.dataframe(df, height=400)  # height in pixels, adjust as needed


# Select Features
features = st.multiselect("Select Features for Clustering", df.columns.tolist(), default=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

if len(features) >= 2:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering_method = st.radio("Choose Clustering Algorithm", ["KMeans", "Hierarchical"])

    if clustering_method == "KMeans":
        # Elbow plot
        st.subheader("📉 Elbow Method for Optimal Clusters")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        # Choose K and cluster
        k = st.slider("Select Number of Clusters (K)", 2, 10, 5)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

        # Show cluster plot
        st.subheader("📊 KMeans Clustering Result")
        fig2 = sns.pairplot(df, hue='KMeans_Cluster', vars=features, palette='tab10')
        st.pyplot(fig2)

    elif clustering_method == "Hierarchical":
        st.subheader("🌳 Dendrogram (Hierarchical Clustering)")

        linked = linkage(X_scaled, method='ward')
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
        st.pyplot(fig3)

        k = st.slider("Select Number of Clusters", 2, 10, 5)
        df['Hierarchical_Cluster'] = fcluster(linked, k, criterion='maxclust')

        st.subheader("📊 Hierarchical Clustering Result")
        fig4 = sns.pairplot(df, hue='Hierarchical_Cluster', vars=features, palette='tab10')
        st.pyplot(fig4)

else:
    st.warning("Please select at least two features to perform clustering.")
