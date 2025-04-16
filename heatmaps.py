import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your table
df = pd.read_csv('host_appearances.csv')  # Change the filename if needed

# 2. Select the year you want 
selected_year = 2021
df_year = df[df['Year'] == selected_year]



# # Optional Filter only for the species of interest
# species_of_interest = [
#     'Chicken', 'Duck', 'Goose', 'Other avian', 'Turkey', 'Gull', 'Swan', 'Eagle',
#     'Wild bird / Wild birds', 'Falcon', 'Black-headed gull', 'Pheasant', 'Meleagris gallopavo',
#     'Crow', 'Seabird', 'Guineafowl', 'Penguin', 'Mallard', 'Shorebird', 'Pigeon', 'Passerine'
# ]

#df_year = df_year[df_year['Host'].isin(species_of_interest)]


# 3. Pivot the table
pivot = df_year.pivot_table(index='Location', columns='Host', values='Count', fill_value=0)


# Convert float to int 
pivot = pivot.astype(int)

pivot.index = pivot.index.str.title()

#  Add a total sum column and sort
pivot['Total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('Total', ascending=False)

#  Keep only the top x countries
pivot = pivot.head(20)

#  Remove the 'Total' column (optional for clean heatmap)
pivot = pivot.drop(columns='Total')

# 4. Plot the heatmap
plt.figure(figsize=(18, 12))  # Adjust size if you want
sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.5, linecolor='gray', annot=True, fmt='d')

plt.title(f'Host Appearances per Country Top 20 - January - March {selected_year}', fontsize=20)
plt.xlabel('Host Species', fontsize=16)
plt.ylabel('Country', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(fontsize=10)
plt.tight_layout()

plt.savefig('heatmap_species_country_(year).png', dpi=300, bbox_inches='tight') #change file name as needed 

plt.show()