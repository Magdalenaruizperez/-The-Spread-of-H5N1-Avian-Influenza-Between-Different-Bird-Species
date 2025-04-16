# ------------------------
# Analysis #2: Network Map of Species and Countries
# ------------------------


#needs cleaning on validation - new code and try each
import random
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ast
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from tqdm import tqdm
import seaborn as sns 
from networkx.algorithms import community  # For community detection

#Colours
#Red Shades: [ #FF0000 - Red , #FF6347 - Tomato , #DC143C - Crimson , #B22222 - Firebrick , #FF4500 - Orange Red ]
#Blue Shades: [ #0000FF - Blue , #4682B4 - Steel Blue , #1E90FF - Dodger Blue , #5F9EA0 - Cadet Blue , #ADD8E6 - Light Blue ]
#Green Shades: [ #008000 - Green , #32CD32 - Lime Green , #98FB98 - Pale Green , #2E8B57 - Sea Green , #228B22 - Forest Green ]
#Yellow Shades: [ #FFFF00 - Yellow , #FFD700 - Gold , #FFFACD - Lemon Chiffon , #F0E68C - Khaki , #ADFF2F - Green Yellow ]
#Orange Shades: [ #FFA500 - Orange , #FF8C00 - Dark Orange , #FF7F50 - Coral , #FF6347 - Tomato , #FF4500 - Orange Red ]
#Purple Shades: [ #800080 - Purple , #8A2BE2 - Blue Violet , #9370DB - Medium Purple , #D8BFD8 - Thistle , #9932CC - Dark Orchid ]
#Pink Shades: [ #FFC0CB - Pink , #FF69B4 - Hot Pink , #FF1493 - Deep Pink , #DB7093 - Pale Violet Red , #C71585 - Medium Violet Red ]
#Brown Shades: [ #A52A2A - Brown , #D2691E - Chocolate , #8B4513 - Saddle Brown , #A52A2A - Brown , #F4A300 - Dark Goldenrod ]
#Gray Shades: [ #808080 - Gray , #A9A9A9 - Dark Gray , #D3D3D3 - Light Gray , #BEBEBE - Gray 80% , #2F4F4F - Dark Slate Gray ]
#Black and White Shades: [ #000000 - Black , #FFFFFF - White , #F8F8FF - Ghost White , #DCDCDC - Gainsboro , #696969 - Dim Gray ]
#Teal and Aqua: [ #008080 - Teal , #00FFFF - Aqua , #20B2AA - Light Sea Green , #40E0D0 - Turquoise , #5F9EA0 - Cadet Blue ]
#Beige and Tan: [ #F5F5DC - Beige , #D2B48C - Tan , #F4A460 - Sandy Brown , #C2B280 - Khaki , #FFDEAD - Navajo White ]




#  1. Load data
data = pd.read_csv("host_location.csv")

# Capitalize country names for neatness
data['Location'] = data['Location'].str.title()


# 2. Create Graph
G = nx.Graph()



for _, row in data.iterrows():
    G.add_edge(row["Host"], row["Location"])

# 5. Count how many countries each species appears in
species_country_counts = data.groupby('Host')['Location'].nunique().to_dict()

# 6. Assign edge colors based on how many countries the species migrates to
edge_colors = []
for u, v in G.edges():
    if u in species_country_counts:
        species = u
    else:
        species = v

    num_countries = species_country_counts.get(species, 0)



    if num_countries < 5:
        edge_colors.append('#FF9999')  # Light Red (few countries)
    elif 5 <= num_countries <= 10:
        edge_colors.append('#800080')  # Purple (moderate)
    elif 10 < num_countries <= 20:
        edge_colors.append('#FF1493')  # Deep Pink (many)
    else:
        edge_colors.append('#4682B4')  # Steal Blue (very many)




# 7. Draw the network
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G, k=0.1)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.7)

# Draw only country names
country_nodes = set(data['Location'])
country_labels = {node: node for node in G.nodes() if node in country_nodes}
nx.draw_networkx_labels(G, pos, labels=country_labels, font_size=12)

# Draw nodes without labels
nx.draw_networkx_nodes(G, pos, node_size=50)

# 8. Create a custom legend
few_patch = mpatches.Patch(color='#FF9999', label='<5 countries')
moderate_patch = mpatches.Patch(color='#800080', label='5–10 countries')
many_patch = mpatches.Patch(color='#FF1493', label='10–20 countries')
very_many_patch = mpatches.Patch(color='#4682B4', label='>20 countries')
plt.legend(handles=[few_patch, moderate_patch, many_patch, very_many_patch], loc='upper right', fontsize=12)

# 9. Finalize
plt.title("Species-Country Migration Network (Edges Colored by Species Spread)", fontsize=22)
plt.axis('off')
plt.tight_layout()
# plt.savefig("species_country_network.png", dpi=300)

# plt.show()

#### END OF MAIN GRAPH ####


#### CALCULATE METRICS ####

# Calculate Degree Centrality
degree_centrality = nx.degree_centrality(G)
# print("Degree Centrality:", degree_centrality)

# Calculate Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(G)
# print("Betweenness Centrality:", betweenness_centrality)

# Calculate Clustering Coefficient
clustering_coefficient = nx.clustering(G)
# print("Clustering Coefficient:", clustering_coefficient)


# Create DataFrames for each metric
df_degree = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['Degree_Centrality'])
df_betweenness = pd.DataFrame.from_dict(betweenness_centrality, orient='index', columns=['Betweenness_Centrality'])
df_clustering = pd.DataFrame.from_dict(clustering_coefficient, orient='index', columns=['Clustering_Coefficient'])

# Combine into one table
df_combined = pd.concat([df_degree, df_betweenness, df_clustering], axis=1)

# Get species and country lists directly from the metrics (since we don't have G)
all_nodes = list(degree_centrality.keys())
species_nodes = [n for n in all_nodes if n in data['Host'].unique()]  # Assuming 'data' exists
country_nodes = [n for n in all_nodes if n in data['Location'].unique()]

df_species = df_combined.loc[species_nodes]
df_countries = df_combined.loc[country_nodes]

# Export
df_species.to_csv('species_centrality_metrics.csv')
df_countries.to_csv('countries_centrality_metrics.csv')
df_combined.to_csv('all_nodes_centrality_metrics.csv')

print("Exported 3 files:")
print("- species_centrality_metrics.csv")
print("- countries_centrality_metrics.csv")
print("- all_nodes_centrality_metrics.csv")


#### END OF METRICS ####

#### RANDOMNESS AND P VALUES ####


# Number of random networks to generate
num_random_networks = 100

# Store the degree centralities and betweenness centralities of random networks
random_degree_centralities = []
random_betweenness_centralities = []

# Generate random networks and calculate their centralities
for _ in range(num_random_networks):
    random_G = nx.gnm_random_graph(len(G.nodes()), len(G.edges()))  # Random graph with same size
    random_degree_centralities.append(nx.degree_centrality(random_G))
    random_betweenness_centralities.append(nx.betweenness_centrality(random_G))

# Get the list of nodes in the real network
real_nodes = list(G.nodes())

# Map node names to indices
node_to_index = {node: idx for idx, node in enumerate(real_nodes)}

# Check mapping for 'Bean goose'
print("Index of 'Bean goose':", node_to_index['Bean goose'])



# Initialize dictionaries to store p-values
degree_p_values = {}
betweenness_p_values = {}

# Calculate p-values for each node
for node in real_nodes:
    # Get the index of the node in the real network
    node_idx = node_to_index[node]

    # Real degree centrality
    real_degree = degree_centrality[node]
    
    # Real betweenness centrality
    real_betweenness = betweenness_centrality[node]
    
    # Degree centrality p-value (compare to random networks)
    # Use the index to access the degree centrality for this node in random networks
    random_degrees = [r[node_idx] for r in random_degree_centralities]
    p_value_degree = np.mean([degree >= real_degree for degree in random_degrees])
    degree_p_values[node] = p_value_degree
    
    # Betweenness centrality p-value (compare to random networks)
    # Use the index to access the betweenness centrality for this node in random networks
    random_betweennesses = [r[node_idx] for r in random_betweenness_centralities]
    p_value_betweenness = np.mean([betweenness >= real_betweenness for betweenness in random_betweennesses])
    betweenness_p_values[node] = p_value_betweenness




# Prepare your data for export (using the p-values and centrality values you've calculated)
results = {
    'Node': list(G.nodes()),  # List of all nodes in the real network
    'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
    'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes()],
    'Degree P-value': [degree_p_values[node] for node in G.nodes()],
    'Betweenness P-value': [betweenness_p_values[node] for node in G.nodes()]
}

# Create DataFrame
df = pd.DataFrame(results)

# Export the DataFrame to a CSV file
df.to_csv('node_p_values.csv', index=False)

print("Data has been saved to node_p_values.csv.")


# scatter plot ## 

Assuming these variables have already been calculated
degree_centrality, betweenness_centrality, degree_p_values, betweenness_p_values

# Convert dictionaries to lists of values
degree_values = list(degree_centrality.values())
betweenness_values = list(betweenness_centrality.values())

# Convert p-value dictionaries to lists
degree_p_values = list(degree_p_values.values())
betweenness_p_values = list(betweenness_p_values.values())

# Ensure the lengths match
assert len(degree_values) == len(betweenness_values), "Degree and Betweenness centralities must have the same length"

# Create masks for significant nodes (p-value <= 0.05)
significant_degree_mask = np.array(degree_p_values) <= 0.05
significant_betweenness_mask = np.array(betweenness_p_values) <= 0.05

# Combined significance - significant in either degree OR betweenness
significant_mask = significant_degree_mask | significant_betweenness_mask

# Create the scatter plot
plt.figure(figsize=(10, 6))

# Plot non-significant nodes (green) — where p-value > 0.05 for both
plt.scatter(np.array(degree_values)[~significant_mask], 
            np.array(betweenness_values)[~significant_mask], 
            color='green', label='Non-significant', alpha=0.7)

# Plot significant nodes (blue) — where p-value <= 0.05 for either
plt.scatter(np.array(degree_values)[significant_mask], 
            np.array(betweenness_values)[significant_mask], 
            color='blue', label='Significant', alpha=0.7)

# Labels and title
plt.xlabel('Degree Centrality')
plt.ylabel('Betweenness Centrality')
plt.title('Centrality Measures vs. Statistical Significance')

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()













#### COMMUNITY ####




# Creates communities
communities = list(community.louvain_communities(G, seed=42))

# Print the number of communities
print(f"\nFound {len(communities)} communities")

# Loop through each community and print its size (length of species/countries in each community)
for i, comm in enumerate(communities):
    print(f"Community {i+1} has {len(comm)} species/countries.")



# Open a file in write mode
with open('communities_report.txt', 'w') as f:
    f.write(f"Found {len(communities)} communities\n\n")
    
    # Loop through each community and write the full list to the file
    for i, comm in enumerate(communities):
        f.write(f"Community {i+1} has {len(comm)} species/countries:\n")
        f.write(", ".join(comm))  # Join the species/countries into a string
        f.write("\n\n")

print("Full list of species/countries has been saved to 'communities_report.txt'.")



# Step 2: Create colormap for visualization 
cmap = plt.cm.get_cmap('tab20', len(communities))

# Step 3: Calculate the layout for the network graph
pos = nx.spring_layout(G, k=0.15, seed=42)

# Step 4: Create the figure to draw the graph
plt.figure(figsize=(16, 12))

# Step 5: Draw each community with a different color
for i, comm in enumerate(communities):
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=list(comm),
        node_color=[cmap(i)],
        node_size=50,
        edgecolors='black',
        linewidths=0.5,
        label=f'Community {i+1} (n={len(comm)})'
    )

# Step 6: Draw edges (light grey for background connections)
nx.draw_networkx_edges(G, pos, alpha=0.1, width=0.5)

# Step 7: Add the legend for the communities
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    title="Communities",
    fontsize=10
)

# Step 8: Label 5 most central nodes per community
central_nodes = []
for i, comm in enumerate(communities):
    degrees = {n: G.degree(n) for n in comm}
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:5]
    central_nodes.extend(top_nodes)

labels = {n: n for n in central_nodes}
nx.draw_networkx_labels(
    G, pos,
    labels,
    font_size=8,
    font_color='black'
)

# Step 9: Title and final adjustments for display
plt.title(f"Detected Communities (n={len(communities)})", fontsize=14)
plt.axis('off')
plt.tight_layout()
# plt.show()

# After detecting communities, add:
for i, comm in enumerate(communities):
    species = [n for n in comm if n in data['Host'].unique()]
    countries = [n for n in comm if n in data['Location'].unique()]
    print(f"\nCommunity {i+1}:")
    print(f"  Species ({len(species)}): {', '.join(species[:3])}{'...' if len(species)>3 else ''}")
    print(f"  Countries ({len(countries)}): {', '.join(countries[:3])}{'...' if len(countries)>3 else ''}")

communities_as_sets = [set(comm) for comm in communities]

# Compute modularity
modularity = community.modularity(G, communities_as_sets)
print(f"Modularity: {modularity:.3f}")

from networkx.algorithms import community

# Calculate per-community contribution
communities_as_sets = [set(comm) for comm in communities]
total_modularity = community.modularity(G, communities_as_sets)

for i, comm in enumerate(communities_as_sets):
    comm_modularity = community.modularity(G, [comm, set(G.nodes()) - comm])
    print(f"Community {i+1}: {comm_modularity:.3f} contribution")

    # Highlight low-contribution communities
low_mod_communities = [i for i, comm in enumerate(communities) if len(comm) < 5]  # Small communities often weaken modularity

for i, comm in enumerate(communities):
    color = 'red' if i in low_mod_communities else cmap(i)
    nx.draw_networkx_nodes(G, pos, nodelist=comm, node_color=[color])


# #### END OF COMMUNITY ####

# #### TESTS ####


#Parameter Sensitivity Analysis

# Goal: Understand how choices (e.g., seed=42, k=0.15) affect the graph.

#Test layout stability

pos2 = nx.spring_layout(G, k=0.15, seed=123)  # Different seed
pos3 = nx.spring_layout(G, k=0.5, seed=42)     # Different k

# Compare visually
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for i, (p, title) in enumerate(zip([pos2, pos3], ["Seed=123 k=0.15", "Seed=42 k=0.5"])):
    for j, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, p, nodelist=list(comm), node_color=[cmap(j)], ax=axes[i], node_size=50)
    axes[i].set_title(title)
    axes[i].axis('off')
plt.show()




# 3. Community Interpretation
# Goal: Explain why species/countries cluster together.

#Print 3 largest communities with examples

for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:3]):
    species = [n for n in comm if n in data["Host"].unique()]
    countries = [n for n in comm if n in data["Location"].unique()]
    print(f"\nCommunity {i+1} (Size: {len(comm)}):")
    print(f"  Species: {', '.join(species[:3])}...")
    print(f"  Countries: {', '.join(countries[:3])}...")

# 4. Edge/Node Anomalies
#Goal: Identify potential data issues.

#Check isolated nodes (if any)

isolated = list(nx.isolates(G))
print(f"Isolated nodes (missing in image?): {isolated}")

# Check edge weights (if your data had them)
if "Weight" in data.columns:
    print(f"Edge weight range: {data['Weight'].min()} to {data['Weight'].max()}")
 
# ### END OF TESTS ####

# #### VALIDATION ####

def proper_validation(G, data, n_permutations=500):
    """Robust network validation with configuration model"""
    # Get node lists
    species = data['Host'].unique()
    countries = data['Location'].unique()
    
    # Degree sequences
    deg_seq_species = [G.degree[n] for n in species]
    deg_seq_countries = [G.degree[n] for n in countries]
    
    # Observed metrics
    obs_degree = nx.degree_centrality(G)
    obs_betweenness = nx.betweenness_centrality(G, normalized=True)
    
    # Initialize results storage
    results = []
    
    # Permutation test with progress bar
    for _ in tqdm(range(n_permutations), desc="Running validations"):
        # Generate random network
        random_G = bipartite.configuration_model(deg_seq_species, deg_seq_countries)
        random_G = nx.Graph(random_G)  # Convert to simple graph
        random_G.remove_edges_from(nx.selfloop_edges(random_G))
        
        # Calculate random metrics
        random_degree = nx.degree_centrality(random_G)
        random_betweenness = nx.betweenness_centrality(random_G, normalized=True)
        
        # Store comparison results
        for node in G.nodes():
            deg_compare = sum(d >= obs_degree[node] for d in random_degree.values())
            btw_compare = sum(b >= obs_betweenness[node] for b in random_betweenness.values())
            
            results.append({
                'Node': node,
                'Node_Type': 'Species' if node in species else 'Country',
                'Degree_compare': deg_compare,
                'Betweenness_compare': btw_compare,
                'N_perm': len(random_degree)
            })
    
    # Convert to DataFrame and calculate p-values
    results_df = pd.DataFrame(results)
    validation_df = results_df.groupby(['Node', 'Node_Type']).agg({
        'Degree_compare': 'sum',
        'Betweenness_compare': 'sum',
        'N_perm': 'first'
    }).reset_index()
    
    # Calculate proper p-values with continuity correction
    validation_df['Degree_p_value'] = (validation_df['Degree_compare'] + 1) / (validation_df['N_perm'] + 1)
    validation_df['Betweenness_p_value'] = (validation_df['Betweenness_compare'] + 1) / (validation_df['N_perm'] + 1)
    
    # Add observed values
    validation_df['Observed_Degree'] = validation_df['Node'].map(obs_degree)
    validation_df['Observed_Betweenness'] = validation_df['Node'].map(obs_betweenness)
    


    return validation_df


validation_df = proper_validation(G, data, n_permutations=500)

#Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data=validation_df, x='Degree_p_value', bins=20, kde=True)
plt.axvline(0.05, color='red', linestyle='--', label='p=0.05')
plt.title("Degree Centrality p-values")
plt.legend()

plt.subplot(1, 2, 2)
sns.histplot(data=validation_df, x='Betweenness_p_value', bins=20, kde=True)
plt.axvline(0.05, color='red', linestyle='--', label='p=0.05')
plt.title("Betweenness Centrality p-values")
plt.legend()
plt.tight_layout()
plt.show()

#Significant nodes
significant_nodes = validation_df[
    (validation_df['Degree_p_value'] < 0.05) | 
    (validation_df['Betweenness_p_value'] < 0.05)
]

print(f"\nFound {len(significant_nodes)} significant nodes:")
print(significant_nodes.sort_values('Degree_p_value')[['Node', 'Node_Type', 'Degree_p_value', 'Betweenness_p_value']].to_string())

print(f"Degree centrality variance: {validation_df['Observed_Degree'].var():.3f}")
print(f"Betweenness centrality variance: {validation_df['Observed_Betweenness'].var():.3f}")



# Assuming 'G' is your graph and 'Chicken' is the significant node
plt.figure(figsize=(10, 10))

# Set up the position for the nodes
pos = nx.spring_layout(G, seed=123)

# Draw the nodes with different colors
node_color = ['red' if node == 'Chicken' else 'blue' for node in G.nodes()]
node_size = [G.degree(node) * 10 for node in G.nodes()]

# Draw the graph
nx.draw_networkx(G, pos, node_color=node_color, node_size=node_size, with_labels=True)

# Highlight Chicken
plt.title('Network Visualization with Significant Node (Chicken)', fontsize=15)
plt.show()



# #### END VALIDATION ####


print("Real network density:", nx.density(G))


from networkx.algorithms import bipartite


# Number of species and countries
num_species = 32  # 32 species
num_countries = 85  # 82 countries

# Now create the bipartite random graph
random_G = nx.bipartite.random_graph(num_species, num_countries, p=0.1)
print("Random network density:", nx.density(random_G))

sns.histplot(data=validation_df, x='Degree_p_value', bins=20)
plt.axvline(0.05, color='red', linestyle='--')
plt.title("Distribution of Degree p-values")
plt.show()


 #### STATISTICAL TEST EXPORT ####
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# Perform Mann-Whitney U test
migratory_species = set(data[data['Migratory?'] == 'Yes']['Host'])
non_migratory_species = set(data[data['Migratory?'] == 'No']['Host'])
migratory_betweenness = [betweenness_centrality[s] for s in migratory_species if s in betweenness_centrality]
non_migratory_betweenness = [betweenness_centrality[s] for s in non_migratory_species if s in betweenness_centrality]

statistic, p_value = mannwhitneyu(migratory_betweenness, non_migratory_betweenness, alternative='greater')

# Export to CSV
results_df = pd.DataFrame({
    'Test': ['Mann-Whitney U (Migratory vs. Non-Migratory)'],
    'Statistic': [statistic],
    'p-value': [p_value],
    'Conclusion': ['Migratory species have higher betweenness' if p_value < 0.05 
                  else 'No significant difference']
})
results_df.to_csv('mann_whitney_results.csv', index=False)

# Export group summary stats
summary_df = pd.DataFrame({
    'Group': ['Migratory', 'Non-Migratory'],
    'Mean Betweenness': [np.mean(migratory_betweenness), np.mean(non_migratory_betweenness)],
    'Median Betweenness': [np.median(migratory_betweenness), np.median(non_migratory_betweenness)],
    'Sample Size': [len(migratory_betweenness), len(non_migratory_betweenness)]
})
summary_df.to_csv('group_summary_stats.csv', index=False)

print("Statistical results exported to CSV files.")

#### END OF STATISTICAL TEST EXPORT ####

# Boxplot for migratory vs. non-migratory betweenness
import seaborn as sns
sns.boxplot(x=['Migratory']*len(migratory_betweenness) + ['Non-Migratory']*len(non_migratory_betweenness),
            y=migratory_betweenness + non_migratory_betweenness)
plt.title("Betweenness Centrality: Migratory vs. Non-Migratory Species")
plt.ylabel("Betweenness Centrality")
plt.show()









































