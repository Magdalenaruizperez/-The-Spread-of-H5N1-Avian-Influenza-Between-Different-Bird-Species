
import pandas as pd
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


# 1. Load your Excel file
outbreak = pd.read_csv("gisaid_epiflu_isolates_full_period.csv")  # <-- this used GISAID raw data / replace with your file name


# 2. Load world countries shapefile
world = gpd.read_file("C:/Users/maggi/Documents/1 - Bioinformatics/choropleth/world_shapefile/ne_110m_admin_0_countries.shp")

# 3. Adjust the 'Country' field to keep only the first two parts

def adjust_location(location):
    if pd.isna(location):
        return location
    parts = location.split('/')
    if len(parts) >= 2:
        return parts[0].strip() + " / " + parts[1].strip()
    else:
        return location.strip()


outbreak['Location'] = outbreak['Location'].apply(adjust_location)


# 4. Clean case and spaces
outbreak['Location'] = outbreak['Location'].str.strip().str.lower()
world['NAME'] = world['NAME'].str.strip().str.lower()

# 4. Apply corrections to outbreak country names

corrections = {
    'europe / czech republic': 'czechia',
    'europe / netherlands': 'netherlands',
    'europe / sweden': 'sweden',
    'asia / israel': 'israel',
    'europe / belgium': 'belgium',
    'europe / poland': 'poland',
    'europe / croatia': 'croatia',
    'africa / south africa': 'south africa',
    'europe / norway': 'norway',
    'europe / luxembourg': 'luxembourg',
    'asia / japan': 'japan',
    'europe / germany': 'germany',
    'europe / france': 'france',
    'north america / united states': 'united states of america',
    'north america / ': 'united states of america',
    'north america / canada': 'canada',
    'europe / austria': 'austria',
    'europe / montenegro': 'montenegro',
    'europe / russian federation': 'russia',
    'europe / denmark': 'denmark',
    'asia / philippines': 'philippines',
    'south america / peru': 'peru',
    'europe / switzerland': 'switzerland',
    'south america / venezuela, bolivarian republic of': 'venezuela',
    'south america / ecuador': 'ecuador',
    'north america / costa rica': 'costa rica',
    'south america / brazil': 'brazil',
    'europe / united kingdom': 'united kingdom',
    'europe / slovakia': 'slovakia',
    'asia / korea, republic of': 'south korea',
    'antarctica / south georgia and the south sandwich islands': 'united kingdom',
    'asia / vietnam': 'vietnam',
    'antarctica / torgersen island': 'antarctica',
    'europe / italy': 'italy',
    'europe / isle of man': 'united kingdom',
    'south america / chile': 'chile',
    'europe / slovenia': 'slovenia',
    'africa / burkina faso': 'burkina faso',
    'europe / jersey': 'united kingdom',
    'south america / argentina': 'argentina',
    'asia / bangladesh': 'bangladesh',
    'asia / china': 'china',
    'europe / finland': 'finland',
    'europe / spain': 'spain',
    'europe / romania': 'romania',
    'europe / moldova, republic of': 'moldova',
    'africa / mauritania': 'mauritania',
    'asia / iraq': 'iraq',
    'asia / indonesia': 'indonesia',
    'asia / india': 'india',
    'oceania / australia': 'australia',
    'africa / ghana': 'ghana',
    'africa / egypt': 'egypt',
    "asia / lao, people's democratic republic": 'laos',
    'south america / uruguay': 'uruguay',
    'europe / bulgaria': 'bulgaria',
    'africa / gambia': 'gambia',
    'europe / serbia': 'serbia',
    'europe / cyprus': 'cyprus',
    'asia / kazakhstan': 'kazakhstan',
    'south america / falkland islands (islas malvinas)': 'united kingdom',
    'europe / iceland': 'iceland',
    'africa / niger': 'niger',
    'south america / bolivia': 'bolivia',
    'north america / mexico': 'mexico',
    'antarctica / antarctica': None,
    'africa / nigeria': 'nigeria',
    'europe / lithuania': 'lithuania',
    'europe / estonia': 'estonia',
    'north america / puerto rico': 'united states of america',
    'south america / colombia': 'colombia',
    'europe / ireland': 'ireland',
    'europe / greece': 'greece',
    'europe / albania': 'albania',
    'asia / cambodia': 'cambodia',
    'north america / guatemala': 'guatemala',
    'asia / yemen': 'yemen',
    'asia / maldives': 'maldives',
    'africa / somalia': 'somalia',
    'europe / hungary': 'hungary',
    'europe / portugal': 'portugal',
    'europe / latvia': 'latvia',
    'europe / svalbard and jan mayen': 'norway',
    'north america / honduras': 'honduras',
    'africa / namibia': 'namibia',
    'north america': None,
    'north america / panama': 'panama',
    'asia / hong kong (sar)': 'china',
    'africa / mali': 'mali',
    'europe / macedonia, the former yogoslav republic of': 'north macedonia',
    
    # NEW corrections for the missing ones
    'asia / georgia': 'georgia',
    'africa / senegal': 'senegal',
    'africa / botswana': 'botswana',
    'africa / lesotho': 'lesotho',
    'europe / bosnia and herzegovina': 'bosnia and herzegovina',
    'africa / benin': 'benin',
}




outbreak['Location'] = outbreak['Location'].replace(corrections)

# 5. Drop rows where Country is still missing (NaN)
outbreak = outbreak.dropna(subset=['Location'])

# 6. OPTIONAL: Check if any countries still do not match
missing_countries = outbreak[~outbreak['Location'].isin(world['NAME'])]['Location'].unique()
print("❌ Missing countries after corrections:")
print(missing_countries)


mapping = {
    

    #Meleagris gallopavo

    'Meleagris gallopavo':'Meleagris gallopavo',

    # Chickens
    'Avian' : 'Chicken',
    'Gallus gallus': 'Chicken',
    'Gallus gallus domesticus': 'Chicken',
    'Poultry': 'Chicken',

    # Ducks
    'Mallard duck' : 'Duck',
    'Cairina moschata' : 'Duck',
    'Anas platyrhynchos f. domestica' : 'Duck',
    'Anas platyrhynchos': 'Duck',
    'Anas platyrhynchos var. domesticus': 'Duck',
    'Anas sp.': 'Duck',
    'Dabbling duck': 'Duck',
    'Common teal': 'Duck',
    'Anas penelope': 'Duck',
    'Anas crecca': 'Duck',
    'Anas strepera': 'Duck',
    'Anas cyanoptera': 'Duck',
    'Anas zonorhyncha': 'Duck',
    'Aythya fuligula': 'Duck',
    'Aythya affinis': 'Duck',
    'Northern shoveler': 'Duck',
    'Mareca penelope': 'Duck',
    'Tadorna tadorna': 'Duck',
    'Somateria mollissima': 'Duck',
    'Lophodytes cucullatus': 'Duck',
    'Anseriformes sp.': 'Duck',
    'Teal': 'Duck',
    'Aix galericulata' : 'Duck',




    # Geese
    'Anser indicus' : 'Goose',
    'Anser anser': 'Goose',
    'Anser anser domesticus': 'Goose',
    'Greylag goose': 'Goose',
    'Anser brachyrhynchus': 'Goose',
    'Branta canadensis': 'Goose',
    'Branta leucopsis': 'Goose',
    'Branta bernicla': 'Goose',
    'Anser albifrons': 'Goose',
    'Anser caerulescens': 'Goose',
    'Anser rossii': 'Goose',
    'White-fronted goose': 'Goose',
    'Domestic goose': 'Goose',

    # Swans
    'Cygnus olor': 'Swan',
    'Cygnus cygnus': 'Swan',
    'Cygnus columbianus': 'Swan',
    'Cygnus atratus': 'Swan',
    'Cygnus melancoryphus': 'Swan',
    'Whooper swan': 'Swan',

    # Gulls
    'Gull': 'Gull',
    'Larus argentatus': 'Gull',
    'Larus marinus': 'Gull',
    'Herring gull': 'Gull',
    'Larus ridibundus': 'Gull',
    'Larus canus': 'Gull',
    'Larus': 'Gull',
    'Larus fuscus': 'Gull',
    'Larus dominicanus': 'Gull',
    'Larus cachinnans': 'Gull',
    'Larus smithsonianus': 'Gull',
    'Larus melanocephalus': 'Gull',
    'Larus delawarensis': 'Gull',
    'Chroicocephalus ridibundus': 'Gull',
    'Chroicocephalus': 'Gull',
    'Leucophaeus': 'Gull',

    # Falcons
    'Falcon': 'Falcon',
    'Falco peregrinus': 'Falcon',
    'Falco tinnunculus': 'Falcon',
    'Falco': 'Falcon',

    # Eagles
    'Eagle': 'Eagle',
    'Halietus albicilla': 'Eagle',
    'Haliaeetus leucocephalus': 'Eagle',
    'Nisaetus nipalensis': 'Eagle',
    'Buteo buteo': 'Eagle',
    'Buteo jamaicensis': 'Eagle',
    'Buteo lineatus': 'Eagle',
    'Accipiter gentilis': 'Eagle',
    'Accipiter nisus': 'Eagle',

    # Pigeons/Doves
    'Pigeon': 'Pigeon',
    'Dove': 'Pigeon',

    # Cormorants
    'Cormorant': 'Cormorant',

    # Penguins
    'Penguin': 'Penguin',

    # Other birds
    'Crow': 'Crow',
    'Passerine': 'Passerine',
    'Partridge': 'Partridge',
    'Quail': 'Quail',
    'Pheasant': 'Pheasant',
    'Phasianus colchicus': 'Pheasant',
    'Phasanius colchicus': 'Pheasant',
    'Pica': 'Crow',

    # Seabirds
    'Sterna hirundo': 'Seabird',
    'Sterna sandvicensis': 'Seabird',
    'Sterna paradisaea': 'Seabird',
    'Rissa tridactyla': 'Seabird',
    'Rynchops niger': 'Seabird',
    'Larosterna inca': 'Seabird',
    'Shearwater': 'Seabird',

    # Other waterbirds
    'Wild bird': 'Wild bird / Wild birds',
    'Wild birds': 'Wild bird / Wild birds',
    'Wild waterfowl': 'Wild bird / Wild birds',
    'Calidris alba': 'Shorebird',
    'Calidris canutus': 'Shorebird',
    'Eurasian curlew': 'Shorebird',
    'Numenius arquata': 'Shorebird',
    'Curlew': 'Shorebird',

    # Others

    'Numida meleagris': 'Guineafowl',
    'Ostrich': 'Ostrich',
    'Sacred ibis': 'Sacred ibis',
    'Guinea fowl': 'Guineafowl',
    'Guineafowl': 'Guineafowl',
    'Gallinula chloropus': 'Waterfowl',
    'Podiceps cristatus': 'Grebe',
    'Tachybaptus ruficollis': 'Grebe',
    'Coturnix sp.': 'Quail',
    'Peafowl': 'Peafowl',
    'Meleagris gallopavo': 'Turkey',
    'Pavo cristatus' : 'Peafowl',
    'Phasanius sp.' : 'Pheasant',
    'Phasianus' : 'Pheasant'
}

# print(df['Host'].unique())
# print(df['Host'].nunique())


# Replace host names according to the mapping
outbreak['Host'] = outbreak['Host'].replace(mapping)


# # 2. Group by Host and collect the list of unique countries
host_countries = outbreak.groupby('Host')['Location'].unique().reset_index()

# # 3. Rename the columns nicely
# host_countries.columns = ['Host', 'Countries']


# 4. Now explode: one row per (Host, Location)
host_countries = host_countries.explode('Location').reset_index(drop=True)


# 4. Define your list of migratory species



migratory_species = ["Duck", "Goose", "Gull", "Swan", "Eagle", 'Wild bird / Wild birds',
"Falcon",'Black-headed gull', 'Seabird', 'Penguin', 'Mallard', 'Shorebird','Passerine',
'Cormorant','Grebe','Quail','Dove','Sacred ibis','Waterfowl']


# 5. Add the 'Migratory' column (Yes/No)
host_countries['Migratory'] = host_countries['Host'].apply(
    lambda x: 'Yes' if x in migratory_species else 'No'
)

# 6. Save to a new CSV file
host_countries.to_csv("host_location.csv", index=False)



print(" New file created successfully!")


#### ANALYSIS #####



#Analysis 1: Count how many countries each species appears in
# This shows:

#Chicken appears in 60+ countries, Falcon appears in 10+, etc.

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your expanded table
df = pd.read_csv('host_location.csv')

# Capitalize the first letter of each word in 'Country'
df['Location'] = df['Location'].str.title()


# 2. Group by Host and count how many unique countries
species_country_count = df.groupby('Host')['Location'].nunique().reset_index(name='Country Count')

# 3. Sort for nicer plot
species_country_count = species_country_count.sort_values('Country Count', ascending=False)

4. Plot
plt.figure(figsize=(14, 7))
plt.barh(species_country_count['Host'], species_country_count['Country Count'], color='salmon')
plt.xlabel('Number of Countries')
plt.title('Number of Countries Each Species Appears In')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Number of Countries Each Species Appears In.png', dpi=300, bbox_inches='tight')

plt.show()


#Analysis 2: Top 10 species with widest spread

# 1. Take the previous species_country_count
# top10_species = species_country_count.head(10)

# 2. Plot
plt.figure(figsize=(12, 6))
plt.barh(top10_species['Host'], top10_species['Country Count'], color='darkorange')
plt.xlabel('Number of Countries')
plt.title('Top 10 Widest Spread Species')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Top 10 Widest Spread Species.png', dpi=300, bbox_inches='tight')

plt.show()


#Analysis 3: Top 10 countries hosting the most different species

# 1. Group by Country and count unique species
country_species_count = df.groupby('Location')['Host'].nunique().reset_index(name='Species Richness')

# 2. Sort
top10_countries = country_species_count.sort_values('Species Richness', ascending=False).head(10)

3. Plot
plt.figure(figsize=(12, 6))
plt.barh(top10_countries['Location'], top10_countries['Species Richness'], color='salmon')
plt.xlabel('Number of Unique Species')
plt.title('Top 10 Countries by Species Richness')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Top 10 Countries by Species Richness.png', dpi=300, bbox_inches='tight')
plt.show()


# Analysis 4: Species richness per country (Heatmap)

import seaborn as sns

# #1. Create pivot table
pivot = df.pivot_table(index='Location', values='Host', aggfunc='nunique')

# 2. Sort
pivot = pivot.sort_values('Host', ascending=False)


# Keep only the top x countries
pivot = pivot.head(20)

# 3. Plot
plt.figure(figsize=(8, 20))
sns.heatmap(pivot, annot=True, cmap='YlOrRd', linewidths=0.5)

#plt.title('Bird vs Bird Heatmap')

# plt.xlabel('Species Count')
# plt.ylabel('Country')
plt.tight_layout()

plt.savefig('Bird vs Bird heatmap.png', dpi=300, bbox_inches='tight')

plt.show()


# --- Species Similarity Heatmap ---

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv("host_location.csv")  

# 2. Create a Pivot Table: Species (rows) x Countries (columns)
pivot = pd.pivot_table(df, 
                       index='Host', 
                       columns='Location', 
                       aggfunc='size', 
                       fill_value=0)

# 3. Compute Species-Species Correlation Matrix
species_corr = pivot.corr(method='pearson')  # or you can use .T.corr() depending on how you set the pivot

# Actually better to do .dot(pivot.T) to compare species properly:
species_similarity = pivot.dot(pivot.T)

# 4. Plot the Heatmap
plt.figure(figsize=(18, 15))
sns.heatmap(species_similarity, cmap="Reds", annot=False, square=True, linewidths=0.5, linecolor='gray')

plt.title("Species Similarity Based on Country Appearance", fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Optional save
plt.savefig("species_similarity_heatmap.png", dpi=300)

plt.show()






# --- MIGRATION ANALYSIS  ---

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 1. Load the cleaned expanded table (Species - Country - Migratory?)
data = pd.read_csv("C:/Users/maggi/Documents/1 - Bioinformatics/choropleth/GISAID/migration/analysis by host location/host_location.csv")

# Capitalize country names for neatness
data['Location'] = data['Location'].str.title()

# ------------------------
# Analysis #1: Species Migration vs Outbreak Richness
# ------------------------

Count how many countries each species appears in
species_country_counts = data.groupby(['Host', 'Migratory?'])['Location'].nunique().reset_index()
species_country_counts.rename(columns={'Location': 'Num_Countries'}, inplace=True)

Plot Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data=species_country_counts, x='Migratory?', y='Num_Countries', palette='Reds')
plt.title("Species Migration vs Geographic Spread", fontsize=16)
plt.ylabel("Number of Countries")
plt.xlabel("Migratory Status")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("migration_vs_spread_boxplot.png", dpi=300)
plt.show()






# ------------------------
# Analysis #3: Cluster Analysis (Dendrogram)
# ------------------------

Create species-country presence matrix
species_country_matrix = pd.crosstab(data['Host'], data['Location'])

# Linkage method for clustering
linked = linkage(species_country_matrix, method='ward')

# Plot Dendrogram
plt.figure(figsize=(15, 7))
dendrogram(linked, labels=species_country_matrix.index.tolist(), leaf_rotation=90)
plt.title("Species Clustering Based on Country Spread", fontsize=16)
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("species_dendrogram.png", dpi=300)
plt.show()

# ------------------------
# Analysis #4: Country Similarity Matrix (Heatmap)
# ------------------------

# Create country-species presence matrix
country_species_matrix = pd.crosstab(data['Location'], data['Host'])

# Calculate correlation between countries
corr_matrix = country_species_matrix.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, cmap="Reds", linewidths=0.5)
plt.title("Country Similarity Based on Species", fontsize=18)
plt.tight_layout()
plt.savefig("country_similarity_heatmap.png", dpi=300)
plt.show()

