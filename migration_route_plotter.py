

import geopandas as gpd
import matplotlib.pyplot as plt

#Trying for duck to falcon

# # 1. Set up migration paths
# migration_paths = {
#     "Duck": [("Canada", "Mexico"), ("Northern Europe", "Africa")],
#     "Goose": [("Arctic", "Southern USA"), ("Arctic", "Africa")],
#     "Gull": [("Arctic", "Temperate Coasts")],
#     "Swan": [("Iceland", "United Kingdom"), ("Arctic", "Eastern USA")],
#     "Eagle": [("Central Asia", "Africa"), ("Alaska", "USA")],
#     "Wild bird / Wild birds": [("Europe", "Africa"), ("North America", "Central America")],
#     "Falcon": [("East Asia", "Southern Africa")],
# }


migration_paths = {
    "Duck": [("Canada", "Mexico"), ("Northern Europe", "Africa")],
    "Goose": [("Arctic", "Southern USA"), ("Arctic", "Africa")],
    "Gull": [("Arctic", "Temperate Coasts")],
    "Swan": [("Iceland", "United Kingdom"), ("Arctic", "Eastern USA")],
    "Eagle": [("Central Asia", "Africa"), ("Alaska", "USA")],
    "Falcon": [("East Asia", "Southern Africa")],
    "Seabird": [("Southern Ocean", "Southern Ocean (circumpolar)")],
    "Penguin": [("Antarctica", "Antarctica (seasonal movement)")],
    "Shorebird": [("Arctic Canada", "Argentina")],
    "Passerine": [("Northern USA", "Central America"), ("Europe", "Africa")],
    "Cormorant": [("Europe", "North Africa")],
    "Grebe": [("Western Canada", "Southern California")],
    "Quail": [("Europe", "Africa")],
    "Dove": [("USA", "Mexico"), ("Europe", "Africa")],
    "Sacred ibis": [("Africa Wetlands", "Africa Wetlands (seasonal)")],
    "Waterfowl": [("North America", "Southern USA and Mexico")],
}


# 2. Load your world shapefile
world = gpd.read_file("C:/Users/maggi/Documents/1 - Bioinformatics/choropleth/GISAID/world_shapefile/ne_110m_admin_0_countries.shp")

# 3. Define center coordinates for locations
center_coords = {
    "Canada": (-100, 60),
    "Mexico": (-102, 23),
    "Northern Europe": (10, 55),
    "Africa": (20, 5),
    "Arctic": (0, 80),
    "Southern USA": (-90, 30),
    "Temperate Coasts": (-70, 40),
    "Iceland": (-20, 65),
    "United Kingdom": (0, 52),
    "Eastern USA": (-75, 40),
    "Central Asia": (70, 45),
    "East Asia": (120, 40),
    "Southern Africa": (25, -30),
    "Europe": (10, 50),
    "North America": (-100, 50),
    "Central America": (-90, 15),
    "Alaska": (-150, 65),
    "USA": (-100, 38),
}

# 4. Start plotting
fig, ax = plt.subplots(figsize=(18, 12))
world.plot(ax=ax, color='lightgrey', edgecolor='black')
plt.title("Migration Paths", fontsize=22)
ax.set_xlim([-180, 180])
ax.set_ylim([-90, 90])
plt.axis('off')

# 5. Plot the migrations
#species_to_plot = ["Duck", "Goose", "Gull", "Swan", "Eagle", "Falcon",'Shorebird','Passerine','Cormorant','Grebe','Quail','Dove','Sacred ibis','Waterfowl']
species_to_plot = ["Duck", "Goose", "Gull", "Swan", "Eagle", "Falcon"]
#colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow','black','purple','blue', 'green', 'red', 'purple']
colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']






for idx, species in enumerate(species_to_plot):
    if species in migration_paths:
        paths = migration_paths[species]
        for start, end in paths:
            if start in center_coords and end in center_coords:
                x_start, y_start = center_coords[start]
                x_end, y_end = center_coords[end]

                ax.annotate("",
                            xy=(x_end, y_end), xycoords='data',
                            xytext=(x_start, y_start), textcoords='data',
                            arrowprops=dict(arrowstyle="->", color=colors[idx], linewidth=2),
                            )
                # Optional: Label at start
                ax.text(x_start, y_start, start, fontsize=8, ha='right', va='bottom', color=colors[idx])

# 6. Add a legend manually
for idx, species in enumerate(species_to_plot):
    ax.plot([], [], color=colors[idx], label=species)

ax.legend(title="Species", loc='lower left', fontsize=10, title_fontsize=12)

plt.show()
