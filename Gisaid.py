
#This code finds out how many counts are per date. 
#Exports a new csv file with the location flock size and collection date
#this is important for the plotting code 

#Also exports an csv file with the type of host and 
#the amount of time it appears 
#(animal-counts needs then name cleaning for unique hosts)



import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


# 1. Load your Excel or CSV file
#outbreak = pd.read_csv("raw-data-gisaid_epiflu_isolates.csv")  # or use read_excel() if Excel file .xls

outbreak = pd.read_csv('gisaid_epiflu_isolates_full_period.csv', encoding='latin1')


# 2. Load world countries shapefile
world = gpd.read_file("world_shapefile/ne_110m_admin_0_countries.shp")

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



# 3. Clean case and spaces
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
    'europe / isle of man': 'united kingdom',  # Isle of Man is a Crown dependency
    'south america / chile': 'chile',
    'europe / slovenia': 'slovenia',
    'africa / burkina faso': 'burkina faso',
    'europe / jersey': 'united kingdom',  # Jersey is a Crown dependency
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
    'south america / falkland islands (islas malvinas)': 'united kingdom',  # Falklands are UK territory
    'europe / iceland': 'iceland',
    'africa / niger': 'niger',
    'south america / bolivia': 'bolivia',
    'north america / mexico': 'mexico',
    'antarctica / antarctica': None,  # Antarctica not mapped to a country
    'africa / nigeria': 'nigeria',
    'europe / lithuania': 'lithuania',
    'europe / estonia': 'estonia',
    'north america / puerto rico': 'united states of america',  # US territory
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
    'europe / svalbard and jan mayen': 'norway',  # Administered by Norway
    'north america / honduras': 'honduras',
    'africa / namibia': 'namibia',
    'north america': None,  # Remove or ignore
    'north america / panama': 'panama',
    'asia / hong kong (sar)': 'china',  # Hong Kong is a SAR of China
    'africa / mali': 'mali',
    'europe / macedonia, the former yogoslav republic of': 'north macedonia'
}



outbreak['Location'] = outbreak['Location'].replace(corrections)

# 5. Drop rows where Country is still missing (NaN)
outbreak = outbreak.dropna(subset=['Location'])

# 6. OPTIONAL: Check if any countries still do not match
missing_countries = outbreak[~outbreak['Location'].isin(world['NAME'])]['Location'].unique()
print("❌ Missing countries after corrections:")
print(missing_countries)

#Species mapping 


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



#  Extract Year from 'Collection_Date' 
outbreak['Year'] = pd.to_datetime(outbreak['Collection_Date'], errors='coerce').dt.year

# 10. Group by Host and Year and count how many times each appears (after cleaning)
#animal_counts = outbreak.groupby(['Host', 'Year']).size().reset_index(name='Count')
animal_period = outbreak.groupby(['Host', 'Year']).size().reset_index(name='Count')

print(animal_counts)

# 11. Save to CSV - works for migration_summary 
animal_period.to_csv("full period/Animal_period.csv", index=False)

print("✅ Cleaned animal counts with year saved successfully.")



#  Group by Location, Host, and Year (works for individual heatmaps)
host_appearances = outbreak.groupby(['Location', 'Host', 'Year']).size().reset_index(name='Appearances')

#  Save to a CSV file
host_appearances.to_csv("full period/host_appearances.csv", index=False)

print("✅ Host appearances by country and year saved successfully.")


print("Host appearances by country saved'")


# Group by Location and Collection Date, and count the number of Hosts
flock_size_outbreak = outbreak.groupby(['Location', 'Collection_Date']).agg({'Host': 'count'}).reset_index()

# # Rename the 'Host' column to 'Flock Size'
flock_size_outbreak = flock_size_outbreak.rename(columns={'Host': 'Flock Size'})

# # Save to new CSV - I need this for ploting the choropleth
flock_size_outbreak.to_csv("/flock_size_FP_outbreak.csv", index=False)

print("File created successfully")


### Flock Size fluctuation analysis ###

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load your table
df = pd.read_csv('Animal_period.csv')  # ⬅️ Replace with your file path!

# Pivot: Years as index, Hosts as columns
pivot = df.pivot_table(index='Year', columns='Host', values='Count', aggfunc='sum', fill_value=0)

# Sort Hosts by total count (important for legend)
pivot = pivot[pivot.sum().sort_values(ascending=False).index]

# Set up the color map
# tab20 has 20 colors, so if you have 21 hosts, extend slightly
colors = cm.get_cmap('tab20', 21)  # use 21 distinct colors

# Plot

plt.figure(figsize=(20, 10))
for idx, host in enumerate(pivot.columns):
    plt.plot(pivot.index, pivot[host], marker='o', label=host, color=colors(idx))

plt.title('Yearly Change in Host Flock Size', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Flock Size', fontsize=14)



# Force only full years on X axis
plt.xticks(ticks=pivot.index, labels=pivot.index.astype(int))

# Legend sorted by total counts
plt.legend(title='Hosts', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.grid(True)
plt.tight_layout()

plt.savefig('Yearly Change Host Flock Size_2.png', dpi=300, bbox_inches='tight')


plt.show()


