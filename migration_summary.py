#Migration summary - use in migration_analysis.py#

import pandas as pd

# 1. Load your table (Host, Count, Year)

file_path = ".csv"

outbreak = pd.read_csv(file_path) #Animal_period

# 2. Define Migratory Species List
# migratory_species = [
#     'Duck', 'Goose', 'Gull', 'Swan', 'Eagle', 'Wild bird / Wild birds',
#     'Falcon', 'Black-headed gull', 'Seabird', 'Penguin', 'Mallard',
#     'Shorebird', 'Passerine'
# ]

migratory_species = ["Duck", "Goose", "Gull", "Swan", "Eagle", 'Wild bird / Wild birds',
"Falcon",'Black-headed gull', 'Seabird', 'Penguin', 'Mallard', 'Shorebird','Passerine',
'Cormorant','Grebe','Quail','Dove','Sacred ibis','Waterfowl']

# 3. Step 1: Add Migratory? column
outbreak['Migratory?'] = outbreak['Host'].apply(lambda x: 'Yes' if x in migratory_species else 'No')

# 4. Step 2: Calculate yearly migratory vs non-migratory counts and percentages
summary = (
    outbreak.groupby('Year')
    .apply(lambda df: pd.Series({
        'Total Outbreak Counts': df['Count'].sum(),
        'Migratory Counts': df[df['Migratory?'] == 'Yes']['Count'].sum(),
        'Non-Migratory Counts': df[df['Migratory?'] == 'No']['Count'].sum()
    }))
    .reset_index()
)

summary['% Migratory'] = (summary['Migratory Counts'] / summary['Total Outbreak Counts'] * 100).round(2)
summary['% Non-Migratory'] = (summary['Non-Migratory Counts'] / summary['Total Outbreak Counts'] * 100).round(2)

# 5. Step 3: Separate detailed tables
migratory_table = outbreak[outbreak['Migratory?'] == 'Yes'].copy()
nonmigratory_table = outbreak[outbreak['Migratory?'] == 'No'].copy()

# 6. Step 3.5: List of species appearing per year (only unique species)
species_list = (
    outbreak[['Year', 'Host', 'Migratory?']]
    .drop_duplicates()
    .sort_values(['Year', 'Host'])
    .reset_index(drop=True)
)

# 7. Step 4: Save all outputs
summary.to_csv('migratory_vs_nonmigratory_summary.csv', index=False)
migratory_table.to_csv('migratory_detailed.csv', index=False)
nonmigratory_table.to_csv('nonmigratory_detailed.csv', index=False)
species_list.to_csv('species_per_year_list.csv', index=False)

# Save the full outbreak table with migratory status
outbreak.to_csv('Animals_period_with_migration.csv', index=False)


print("\nAll outputs have been saved!")

print("- migratory_vs_nonmigratory_summary.csv")
print("- migratory_detailed.csv")
print("- nonmigratory_detailed.csv")
print("- Animals_period_with_migration.csv")#?
