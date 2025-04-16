
#Choropleth maps

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import mapclassify as mapclassify

# 1. Load outbreak data 
outbreak = pd.read_csv('file_name.csv', encoding='latin1')

# 2. Load world countries shapefile
world = gpd.read_file("/world_shapefile/ne_110m_admin_0_countries.shp")


# 3. Clean case and spaces
outbreak['Location'] = outbreak['Location'].str.strip().str.lower()
world['NAME'] = world['NAME'].str.strip().str.lower()


# 4. Group by Location, SUM the flock sizes
outbreak_grouped = outbreak.groupby('Location', as_index=False)['Flock Size'].sum()


# Save to CSV
outbreak_grouped.to_csv("total_flocksize_by_country_.csv", index=False)

print('outbreak per country saved')


# 5. Merge grouped outbreak with world map
merged = world.merge(outbreak_grouped, left_on='NAME', right_on='Location', how='left')



# 8. Reproject for web basemap
merged = merged.to_crs(epsg=3857)



# 9. Plot the map
fig, ax = plt.subplots(figsize=(20, 15))


merged.plot(
   column='Flock Size',                # Color based on deaths
   cmap='plasma',                   # Color scheme (plasma)
   scheme='quantiles',   # <-- 🆕 group similar outbreak sizes
   k = 4,
   linewidth=0.2,
   edgecolor='grey',
   legend=True,
   
   legend_kwds={"title": "Total cases"},
   missing_kwds={"color": "lightgrey"},
   ax=ax
)



#ax.set_xlim([-2.3e7, 2.3e7])  # wider x range
#ax.set_ylim([-3e6, 1.8e7])    # from Antarctica to north Greenland

ax.set_xlim([-2.5e7, 2.5e7])   # Keep the width to see full Pacific to Atlantic
ax.set_ylim([-8e6, 1.8e7])     # Go lower south, more negative, fully show Argentina and Antarctic


# 11. Beautify
ax.set_title("Avian Influenza 2021 Outbreak", fontsize=20)
ax.set_axis_off()

# 12. Save figure
plt.savefig("C:/Users/maggi/Documents/1 - Bioinformatics/choropleth/GISAID/split_by_year_/2021/choropleth 2021.png", dpi=300, bbox_inches='tight')
plt.show()
