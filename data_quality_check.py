# #This code is for identifying which countries did not report the data

import pandas as pd

# 1. Load your CSV
df = pd.read_csv("gisaid_epiflu_isolates_full_period.csv", encoding="latin1")  # Adjust filename if needed

# 2. Extract only the Country from the Location column
def extract_country(location):
    if pd.isna(location):
        return None
    parts = location.split('/')
    if len(parts) >= 2:
        return parts[1].strip().lower()  # Assuming continent/country format
    else:
        return location.strip().lower()

df['Country'] = df['Location'].apply(extract_country)

# 3. Group by Country and collect unique values
summary = df.groupby('Location').agg({
    'Animal_Specimen_Source': lambda x: ', '.join(sorted(set(x.dropna()))),
    'Animal_Health_Status': lambda x: ', '.join(sorted(set(x.dropna()))),
    'Domestic_Status': lambda x: ', '.join(sorted(set(x.dropna())))
}).reset_index()

# 4. Save to new CSV
summary.to_csv("Country_Reported_Values.csv", index=False, encoding="utf-8")

print("✅ New file created: 'Missing_data_Country_Reported_Values.csv'")



#Second part of the code
# First output:
# For each column (Animal_Specimen_Source, Animal_Health_Status, Domestic_Status),
# → how many countries reported (submitted) data?

#  Second output:
# → How many countries submitted all three?
# → How many countries submitted incomplete information (i.e., missing 1+ of the 3 fields)?


# 1. Load your cleaned CSV
df = pd.read_csv("Country_Reported_Values.csv")

# 2. Get list of countries
countries = df['Location'].unique()

print(len(countries))

# 3. Analyze for each country if they submitted the fields
submission_status = []

for country in countries:
    country_data = df[df['Location'] == country]
    
    has_specimen = not country_data['Animal_Specimen_Source'].isna().all()
    has_health = not country_data['Animal_Health_Status'].isna().all()
    has_domestic = not country_data['Domestic_Status'].isna().all()
    
    submission_status.append({
        'Country': country,
        'Submitted_Animal_Specimen_Source': has_specimen,
        'Submitted_Animal_Health_Status': has_health,
        'Submitted_Domestic_Status': has_domestic
    })

status_df = pd.DataFrame(submission_status)

# 4. Count summaries
specimen_count = status_df['Submitted_Animal_Specimen_Source'].sum()
health_count = status_df['Submitted_Animal_Health_Status'].sum()
domestic_count = status_df['Submitted_Domestic_Status'].sum()

all_three_count = len(status_df[
    (status_df['Submitted_Animal_Specimen_Source']) &
    (status_df['Submitted_Animal_Health_Status']) &
    (status_df['Submitted_Domestic_Status'])
])

incomplete_count = len(countries) - all_three_count

# 5. Create a final summary DataFrame
summary_df = pd.DataFrame({
    'Reported_All_Three': [all_three_count],
    'Reported_Incomplete': [incomplete_count],
    'Reported_Animal_Specimen_Source': [specimen_count],
    'Reported_Animal_Health_Status': [health_count],
    'Reported_Domestic_Status': [domestic_count]
})

# 6. Save to a new CSV
summary_df.to_csv("Summary_Country_Submissions.csv", index=False)

print("\n✅ Summary table created successfully: 'Summary_Country_Submissions.csv'")
print(summary_df)



# ✅ Summary table created successfully: 'Summary_Country_Submissions.csv'
#    Reported_All_Three  Reported_Incomplete  ...  Reported_Animal_Health_Status  Reported_Domestic_Status
# 0                 504                 1428  ...                            790                       720