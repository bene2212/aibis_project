import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ===== CONFIG =====
N_CLUSTERS = 3
SEGMENT_FEATURES = [
    "duration",
    "credit_amount",
    "age",
    "installment_rate",
    "residence_since",
    "existing_credits",
    "people_liable"
]
SAVE_DIR = "customer_segmentation"

# ===== DATA LOAD =====
print("📥 Lade numerische Testdaten ...")
X_num = pd.read_csv("data/german_numeric.csv")
X_seg = X_num[SEGMENT_FEATURES].copy()

# ===== SCALING =====
print("📊 Skaliere Segmentierungsdaten ...")
scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(X_seg)

# ===== PCA (Dimensionality Reduction) =====
print("🔬 Führe PCA durch ...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_seg_scaled)

# ===== CLUSTERING (im PCA-Raum) =====
print(f"🤖 Führe KMeans Clustering im PCA-Raum durch (k={N_CLUSTERS}) ...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# ===== Silhouette-Check im PCA-Raum =====
K_TEST = [3, 4, 5, 6, 7]
print("🔎 Silhouette-Scores für verschiedene k (PCA-Raum):")
for k in K_TEST:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km_tmp.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels_tmp)
    print(f"k = {k}: Silhouette-Score = {score:.4f}")

# ===== Elbow-Method im PCA-Raum =====
inertias = []
K_RANGE = range(1, 11)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(list(K_RANGE), inertias, marker="o")
plt.xlabel("Number of clusters k")
plt.ylabel("Inertia (SSE, PCA)")
plt.title("Elbow Method for KMeans (PCA-Raum)")
plt.tight_layout()
os.makedirs(SAVE_DIR, exist_ok=True)
plt.savefig(os.path.join(SAVE_DIR, "elbow_plot_pca.png"), dpi=200)
plt.close()

# ===== RESULT DATA WITH CLUSTER =====
print("📋 Erstelle Segmentierungsresultate ...")
X_seg_result = X_seg.copy()
X_seg_result["cluster"] = clusters

cluster_summary = X_seg_result.groupby("cluster").mean()

# ===== Visualisierung: PCA =====
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=clusters, cmap="tab10", alpha=0.7, s=40
)
plt.xlabel("PCA Komponente 1")
plt.ylabel("PCA Komponente 2")
plt.title("KMeans-Cluster im PCA-Raum")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cluster_pca_2d.png"), dpi=200)
plt.close()

# ===== SAVE EVERYTHING =====
print("💾 Speichere Ergebnisse ...")
os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(kmeans, os.path.join(SAVE_DIR, "kmeans_model_pca.pkl"))
joblib.dump(scaler_seg, os.path.join(SAVE_DIR, "segmentation_scaler.pkl"))
X_seg_result.to_csv(os.path.join(SAVE_DIR, "customer_clusters.csv"), index=False)
cluster_summary.to_csv(os.path.join(SAVE_DIR, "cluster_summary.csv"))
np.save(os.path.join(SAVE_DIR, "X_pca.npy"), X_pca)
config = {
    "n_clusters": N_CLUSTERS,
    "segment_features": SEGMENT_FEATURES,
    "pca_components": 2,
    "n_customers": len(X_seg),
}
with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

print("\n🎉 Customer Segmentation erfolgreich abgeschlossen!")
print(f"📁 Ergebnisse gespeichert unter:  {SAVE_DIR}/")
