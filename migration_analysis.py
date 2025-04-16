#THIS CODE HAS DIFFERENT LEVELS OF ANALYSIS COMPLEXITY 


#MORE BASIC 

##Graph: Bar Plot % Migratory vs Non-Migratory (Per Year)

# Uses migratory_vs_nonmigratory_summary.csv, which looks like this:

# Year	Total Outbreak Counts	Migratory Counts	Non-Migratory Counts	% Migratory	% Non-Migratory
# 2021	100	60	40	60%	40%
# 2022	250	180	70	72%	28%


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the summary
summary = pd.read_csv("migratory_vs_nonmigratory_summary.csv")

# 2. Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Bar width
bar_width = 0.4

# Bar positions
r1 = range(len(summary))
r2 = [x + bar_width for x in r1]

# Plot bars
ax.bar(r1, summary['% Migratory'], width=bar_width, label='% Migratory', color='skyblue', edgecolor='black')
ax.bar(r2, summary['% Non-Migratory'], width=bar_width, label='% Non-Migratory', color='salmon', edgecolor='black')

# X-ticks
ax.set_xticks([r + bar_width/2 for r in range(len(summary))])
ax.set_xticklabels(summary['Year'])

# Labels and Title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Migratory vs Non-Migratory Outbreaks by Year', fontsize=14)
ax.legend()

# Grid for better visibility
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig("migratory_vs_non_migratory_barplot.png", dpi=300, bbox_inches='tight')

plt.show()





# #Medium analysis 
# #Graph: Trend Line: Total Outbreaks of Migratory vs Non-Migratory Birds Over Years

# # Instead of percentages, this time we plot the actual total counts (how many outbreaks migratory and non-migratory birds caused each year).

# # This will show trends:

# # Are migratory species getting more outbreaks over time?
# # Are non-migratory species increasing or decreasing?




# 1. Load the summary
summary = pd.read_csv("migratory_vs_nonmigratory_summary.csv")

# 2. Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# 3. Plotting the lines
ax.plot(summary['Year'], summary['Migratory Counts'], marker='o', label='Migratory', color='skyblue', linewidth=2)
ax.plot(summary['Year'], summary['Non-Migratory Counts'], marker='s', label='Non-Migratory', color='salmon', linewidth=2)

# Labels and Title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Outbreak Counts', fontsize=12)
ax.set_title('Trend of Migratory vs Non-Migratory Outbreaks Over Years', fontsize=14)
ax.legend()


# 🛠 Fix: Set x-axis ticks to whole years
ax.set_xticks(summary['Year'])
ax.set_xticklabels(summary['Year'].astype(int))  # in case they are float

# Grid for better visibility
ax.grid(axis='both', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig("migratory_non_migratory_trend.png", dpi=300, bbox_inches='tight')

plt.show()


#Advanced Analysis
#Graph: Species Activity Over Time (Heatmap)

# What This Heatmap Shows:

# Rows = Each Animal specie animal species.
# Columns = Year.
# Colors = Higher counts (darker colors) = species were more active that year.
# Annotations = Exact number of outbreaks per species per year.

#  Insights You Can Get from the Heatmap:
# Example Analysis	Meaning
# See if some species suddenly rise	(Ex: Wild birds have a sudden outbreak peak in 2023?)
# Detect species that disappear	(Ex: Penguins only appear in 2022?)
# See stability	(Ex: Chicken consistent across all years?)
# Compare migratory vs non-migratory	(Ex: Do migratory birds dominate some years?)




# 1. Load the detailed migratory or non-migratory table 

#Migratory
#df = pd.read_csv("migratory_detailed.csv")
#Non-migratory
#df = pd.read_csv("nonmigratory_detailed.csv")

# 2. Create Pivot Table
pivot = df.pivot_table(index='Host', columns='Year', values='Count', aggfunc='sum', fill_value=0)

# 3. Sort Hosts by total counts (optional, but looks cleaner!)
pivot['Total'] = pivot.sum(axis=1)
pivot = pivot.sort_values('Total', ascending=False).drop(columns=['Total'])


# 4. Plot the Heatmap
plt.figure(figsize=(12, 10))

# changes colour - chose one

# Green to blue - best one
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5, linecolor='gray', annot=True, fmt='g')

# yellow to black
# sns.heatmap(pivot, cmap="inferno", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')

# yellow/dark purple
# sns.heatmap(pivot, cmap="viridis", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')

# purple 
# sns.heatmap(pivot, cmap="coolwarm", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')

# red pallete - white for zeros
# sns.heatmap(pivot, cmap="Reds", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')

# orange to red
# sns.heatmap(pivot, cmap="OrRd", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')

# Stylish dark red modern look
# sns.heatmap(pivot, cmap="rocket", linewidths=0.5, linecolor='gray', annot=True, fmt='.0f')


#plt.title("Host Species Outbreaks by Year (✔️ = Migratory)", fontsize=16)
plt.title('Migratory Species Outbreak Activity by Year', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Host Species', fontsize=14)

plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# Save the heatmap
plt.savefig("migratory_heatmap.png", dpi=300, bbox_inches='tight')

plt.show()


#Graph: Cumulative Sum Plot


# 1. Load your yearly summary table
df = pd.read_csv("migratory_vs_nonmigratory_summary.csv")

# 2. Calculate cumulative sums
df['Cumulative Migratory Counts'] = df['Migratory Counts'].cumsum()
df['Cumulative Non-Migratory Counts'] = df['Non-Migratory Counts'].cumsum()

# 3. Plot
plt.figure(figsize=(10, 6))

plt.plot(df['Year'], df['Cumulative Migratory Counts'], marker='o', label='Migratory Species', color='skyblue')
plt.plot(df['Year'], df['Cumulative Non-Migratory Counts'], marker='o', label='Non-Migratory Species', color='salmon')


plt.title('Cumulative Contribution of Migratory vs Non-Migratory Species Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Outbreak Counts')
plt.xticks(df['Year'])  # only whole years
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("Cumulative Contribution of Migratory vs Non-Migratory Species Over Time.png", dpi=300, bbox_inches='tight')

plt.show()
