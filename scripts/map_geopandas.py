import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from shapely.geometry import Point

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

states = geopandas.read_file("data/cb_2018_us_state_20m/cb_2018_us_state_20m.shp")
states = states[states["NAME"] != "Alaska"]
states = states[states["NAME"] != "Hawaii"]
states = states[states["NAME"] != "Puerto Rico"]

chipotle = pd.read_csv("data/chipotle_stores.csv")
chipotle.head(10)


X=chipotle.loc[:,['latitude','longitude']]

n_cluster = len(chipotle["state"].unique())

id_n=30

def k_means_cluster(id_n: int, X: pd.DataFrame) -> None:
    
    kmeans = KMeans(n_clusters=id_n, random_state=0).fit(X)
    
    cluster_centers = kmeans.cluster_centers_
    
    cluster_labels = pd.Series(kmeans.labels_, name='cluster')
    chipotle_clusters = chipotle.join(cluster_labels.to_frame())
    
    cluster_centroids = pd.DataFrame(cluster_centers, columns=["latitude_centroid","longitude_centroid"])
    
    gdf_centroids = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(cluster_centroids.longitude_centroid, cluster_centroids.latitude_centroid))
    
    gdf = geopandas.GeoDataFrame(chipotle_clusters,
        geometry=geopandas.points_from_xy(chipotle_clusters.longitude, chipotle_clusters.latitude))
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    states.plot(ax = ax, edgecolor='black', color='white')
    
    gdf.plot(markersize=10, alpha=0.5, ax=ax, column='cluster')
    
    gdf_centroids.plot(ax=ax, color='red', alpha = 1, marker = '*', markersize=60)
    
    plt.show()
   
def DBSCAN_cluster(eps_int:int, X: pd.DataFrame) -> None:
    
    clustering = DBSCAN(eps=eps_int, min_samples=2).fit(X)
    
    #cluster_centers = clustering.cluster_centers_
    
    cluster_labels = pd.Series(clustering.labels_, name='cluster')
    chipotle_clusters = chipotle.join(cluster_labels.to_frame())
    print(chipotle_clusters["cluster"].value_counts().sort_index())
    
    #cluster_centroids = pd.DataFrame(cluster_centers, columns=["latitude_centroid","longitude_centroid"])
    
   # gdf_centroids = geopandas.GeoDataFrame(
   #     geometry=geopandas.points_from_xy(cluster_centroids.longitude_centroid, cluster_centroids.latitude_centroid))
    
    gdf = geopandas.GeoDataFrame(chipotle_clusters,
        geometry=geopandas.points_from_xy(chipotle_clusters.longitude, chipotle_clusters.latitude))
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    states.plot(ax = ax, edgecolor='black', color='white')
    
    gdf.plot(markersize=10, alpha=0.5, ax=ax, column='cluster')
    
   # gdf_centroids.plot(ax=ax, color='red', alpha = 1, marker = '*', markersize=60)
    
    plt.show()
    
def Agglomerative_cluster(clusters:int, X: pd.DataFrame) -> None:
        
    model = AgglomerativeClustering(n_clusters=clusters, affinity="euclidean", linkage="complete")
    model = model.fit(X)
    
    cluster_labels = pd.Series(model.labels_, name='cluster')
    chipotle_clusters = chipotle.join(cluster_labels.to_frame())
    print(chipotle_clusters["cluster"].value_counts().sort_index())
    #cluster_centroids = pd.DataFrame(cluster_centers, columns=["latitude_centroid","longitude_centroid"])
    
   # gdf_centroids = geopandas.GeoDataFrame(
   #     geometry=geopandas.points_from_xy(cluster_centroids.longitude_centroid, cluster_centroids.latitude_centroid))
    
    gdf = geopandas.GeoDataFrame(chipotle_clusters,
        geometry=geopandas.points_from_xy(chipotle_clusters.longitude, chipotle_clusters.latitude))
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    
    states.plot(ax = ax, edgecolor='black', color='white')
    
    gdf.plot(markersize=10, alpha=0.5, ax=ax, column='cluster')
    
   # gdf_centroids.plot(ax=ax, color='red', alpha = 1, marker = '*', markersize=60)
    
    plt.show()
    
 
    
def k_means_cluster_silhouette(kmax: int, X: pd.DataFrame) -> None:
    
    sil = []
    #kmax = 10

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax+1):
      kmeans = KMeans(n_clusters = k, random_state=42).fit(X)
      labels = kmeans.labels_
      sil.append(silhouette_score(X, labels, metric = 'euclidean'))
    
    print(sil.index(max(sil))+4)
    plt.plot(range(3, kmax+1),sil)
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.show()

k_means_cluster(37, X)

#DBSCAN_cluster(1.5,X)

# k_means_cluster_silhouette(60,X)


#Agglomerative_cluster(35, X)



