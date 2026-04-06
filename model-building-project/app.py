import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# Title
st.title("🧠 Customer Segmentation App")
st.write("Segment customers based on income and spending behavior using K-Means clustering.")

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data.csv")

# Preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Required columns
required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']

# Validation
if not all(col in df.columns for col in required_columns):
    st.error("CSV must contain 'Annual Income (k$)' and 'Spending Score (1-100)'")
    st.stop()

# Features
X = df[required_columns]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster selection
k = st.slider("Select number of clusters", 2, 10, 5)

# Model
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Show clustered data
st.subheader("📌 Clustered Data")
st.dataframe(df)

# Visualization
st.subheader("📈 Customer Segmentation")

fig, ax = plt.subplots()

ax.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df['Cluster'],
    cmap='viridis'
)

# Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=150,
    c='red',
    marker='X',
    label='Centroids'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score")
ax.legend()

st.pyplot(fig)

# Cluster Insights
st.subheader("🧠 Cluster Insights")

summary = df.groupby('Cluster')[required_columns].mean()
st.dataframe(summary)

st.write("""
- High Income & High Spending → Premium customers  
- Low Income & Low Spending → Budget customers  
- High Income & Low Spending → Potential customers  
- Low Income & High Spending → Impulsive customers  
""")

# 🔥 Elbow Method
st.subheader("📉 Elbow Method (Optimal Clusters)")

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig2, ax2 = plt.subplots()
ax2.plot(range(1, 11), wcss, marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("WCSS")
ax2.set_title("Elbow Method")

st.pyplot(fig2)

# 🔥 Cluster Distribution
st.subheader("📊 Customers per Cluster")

cluster_counts = df['Cluster'].value_counts().sort_index()

fig3, ax3 = plt.subplots()
ax3.bar(cluster_counts.index, cluster_counts.values)
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Number of Customers")

st.pyplot(fig3)

# 🔥 Cluster Filter
st.subheader("🔍 Explore Specific Cluster")

selected_cluster = st.selectbox("Select a cluster", sorted(df['Cluster'].unique()))

filtered_df = df[df['Cluster'] == selected_cluster]

st.write(f"Customers in Cluster {selected_cluster}")
st.dataframe(filtered_df)

# Download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Results", csv, "clustered_data.csv", "text/csv")