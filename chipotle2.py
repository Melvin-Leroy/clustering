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
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gp
from descartes import PolygonPatch

# from reduction import reduce_unique_val

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

# plot 
world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
world.plot()
#plt.show()

#cities = gp.read_file(gp.datasets.get_path('naturalearth_cities'))
gp.datasets.available
#['naturalearth_cities', 'naturalearth_lowres', 'nybb']

ax = world[world.continent == 'North America'].plot(figsize=(10,10), color='white', edgecolor='black')
gdf.plot(ax=ax, color='red')
plt.show()


states = geopandas.read_file('usa-states-census-2014.shp')
type(states)
states.plot()


#states.plot(cmap='magma', figsize=(12, 12))
#states[states['NAME'] == 'Texas']
#states[states['NAME'] == 'Texas'].plot(figsize=(12, 12))


#sans boundaries pour les states
#ax=states.plot()

#avec boundaries pour les states
ax=states.boundary.plot()
gdf.plot(ax=ax, color='red')
plt.show()
























































