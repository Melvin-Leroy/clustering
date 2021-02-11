import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from shapely.geometry import Point

from sklearn.cluster import KMeans

states = geopandas.read_file("Data/cb_2018_us_state_20m/cb_2018_us_state_20m.shp")
states = states[states["NAME"] != "Alaska"]
states = states[states["NAME"] != "Hawaii"]
states = states[states["NAME"] != "Puerto Rico"]

chipotle = pd.read_csv("Data/chipotle_stores.csv")
chipotle.head(10)

'''
gdf = geopandas.GeoDataFrame(
    geometry=geopandas.points_from_xy(chipotle.longitude, chipotle.latitude))

fig, ax = plt.subplots(figsize=(10,10))

states.plot(ax = ax, edgecolor='black', color='white')

gdf.plot(ax=ax, color='red', alpha = 0.1)

plt.show()
'''

X=chipotle.loc[:,['latitude','longitude']]

n_cluster = len(chipotle["state"].unique())

id_n=n_cluster
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

gdf_centroids.plot(ax=ax, color='red', alpha = 1)

plt.show()