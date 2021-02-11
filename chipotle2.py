#Chipotle challenge (piment fumé), donné le 10/02/2021, à rendre pour vendredi 12/02/2021 13h30.
#Find Chipotle epicentres to live your ideal Chipotle lifestyle by clustering the Chipotle dataset
#dataset: chipotle_stores.csv
#State, location, address, latitude, longitude
#map visualization of Chipotles using geopandas (USA)
#visualization of the different clusters
#use Euclidian distance
#exec(open('chipotle2.py').read())


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display 
import geopandas as gp
from descartes import PolygonPatch


df = pd.read_csv("chipotle_stores.csv")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df.head(20)

#OK
df.isnull().sum()

#GeoDataFrame
gdf = gp.GeoDataFrame(df, geometry = gp.points_from_xy(df.longitude, df.latitude))
gdf.head()
#gdf.geometry

#plot 
world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
world.plot()
#plt.show()

#cities = gp.read_file(gp.datasets.get_path('naturalearth_cities'))
gp.datasets.available
#['naturalearth_cities', 'naturalearth_lowres', 'nybb']

ax = world[world.continent == 'North America'].plot(figsize=(10,10), color='white', edgecolor='black')
gdf.plot(ax=ax, color='red')

states = gp.read_file('usa-states-census-2014.shp')
type(states)
states.plot()

#sans boundaries pour les states
#ax=states.plot()

#avec boundaries pour les states
ax=states.boundary.plot()
gdf.plot(ax=ax, color='red')


position = gdf['latitude'].to_frame().join(gdf['longitude'].to_frame())
#normalisation
posnorm = (position-position.mean())/position.std()
posnormstate = gdf['state'].to_frame().join(posnorm)

#scatter plot 
fig=plt.figure(figsize=(10,10))
plt.scatter(posnorm['latitude'], posnorm['longitude'])
plt.title("Location")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.show()


#one colour per state
sns.lmplot(x='latitude', y='longitude', height=10, data=posnormstate, fit_reg=False, hue='state')
plt.show()


#k-means algorithm
#10 clusters
nc = 10
#fit a kmeans object to the dataset (normalized)
kmeans = KMeans(n_clusters=nc, init='k-means++').fit(posnorm)

#attributes
cluster_centers = kmeans.cluster_centers_
cluster_labels = pd.Series(kmeans.labels_, name='cluster')

#dendogram
#ydist=posnorm.to_numpy()
#non normalized
ydist=position.to_numpy()

Z = hierarchy.linkage(ydist, 'single', optimal_ordering = False)

plt.figure()
dn = hierarchy.dendrogram(Z,truncate_mode='level', p=10)

plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()























































