# Script to extract 2020 CELEX numbers from celex_queries and create a new searches_2020.csv file
import pandas as pd
import glob
from pathlib import Path
import re

project_root = Path(__file__).parents[1]
celex_queries_path = project_root / "text_processing" / "celex_queries"
random_sample_path = project_root / "text_processing" / "random_sample.csv"
output_path = project_root / "text_processing" / "searches_2020.csv"

# Function to extract CELEX IDs for 2020 from a dataframe
def extract_2020_celex_ids(df):
    if 'CELEX number' in df.columns:
        # Filter for 2020 CELEX numbers (starting with 32020)
        return df[df['CELEX number'].astype(str).str.startswith('32020')]['CELEX number'].tolist()
    
    # Handle different column names in different files
    celex_col = None
    date_col = None
    
    for col in df.columns:
        if 'CELEX' in col:
            celex_col = col
        if 'Date' in col:
            date_col = col
    
    if celex_col and date_col:
        # Filter by date column for 2020 if it exists
        return df[df[date_col].astype(str).str.startswith('2020')][celex_col].tolist()
    elif celex_col:
        # If no date column, try to extract from CELEX format (3YYYY...)
        return [celex for celex in df[celex_col].astype(str) if celex.startswith('32020')]
    
    return []

# Collect all 2020 CELEX IDs
all_2020_celex_ids = set()

# From celex_queries directory
query_files = glob.glob(str(celex_queries_path / "*.csv"))
for file_path in query_files:
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        celex_ids = extract_2020_celex_ids(df)
        all_2020_celex_ids.update(celex_ids)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# From random_sample.csv
try:
    random_sample_df = pd.read_csv(random_sample_path)
    # Extract the CELEX ID without the 'EU_' prefix
    celex_ids = [doc.replace('EU_', '') for doc in random_sample_df['Document'] 
                 if isinstance(doc, str) and doc.startswith('EU_32020')]
    all_2020_celex_ids.update(celex_ids)
except Exception as e:
    print(f"Error processing random_sample.csv: {e}")

# Create a new DataFrame with the 2020 CELEX IDs
df_2020 = pd.DataFrame({"CELEX number": list(all_2020_celex_ids)})

# Save to searches_2020.csv
df_2020.to_csv(output_path, index=False)

print(f"Found {len(all_2020_celex_ids)} CELEX IDs from 2020")
print(f"Saved to {output_path}")
print("\nList of 2020 CELEX IDs:")
for celex_id in sorted(all_2020_celex_ids):
    print(f"  - {celex_id}")
