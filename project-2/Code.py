import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select key features
df = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Remove outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Boxplot after cleaning
plt.figure(figsize=(8, 4))
sns.boxplot(data=df_clean)
plt.title("Boxplot After Outlier Removal")
plt.tight_layout()
plt.show()

# Scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean)

# Elbow Method
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

# Apply KMeans (set k based on elbow, e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(scaled_data)

# Silhouette Score
score = silhouette_score(scaled_data, df_clean['Cluster'])
print(f"Silhouette Score: {score:.3f}") 

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
plt.scatter(components[:, 0], components[:, 1], c=df_clean['Cluster'], cmap='Set2', s=60)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Optimized Customer Clusters")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# Cluster Summary
cluster_summary = df_clean.groupby('Cluster').mean().round(2)
print("Cluster Summary:\n", cluster_summary)

# Export
cluster_summary.to_csv("optimized_cluster_summary.csv")
df_clean.to_csv("optimized_clustered_customers.csv", index=False)
