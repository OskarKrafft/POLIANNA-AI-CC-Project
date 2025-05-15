# Script to check if 2020 documents from random_sample.csv exist in the processed directories
import pandas as pd
from pathlib import Path

# Load the random sample
random_sample_path = Path(__file__).parents[1] / 'text_processing' / 'random_sample.csv'
random_sample_df = pd.read_csv(random_sample_path)

# Extract 2020 documents from the random sample
docs_2020 = random_sample_df[random_sample_df['Date of document'].str.startswith('2020', na=False)]
celex_ids_2020 = docs_2020['Document'].tolist()

print(f"Found {len(celex_ids_2020)} documents from 2020 in the random sample:")
for celex_id in celex_ids_2020:
    print(f"  - {celex_id}")

# Check if these documents exist in the processed directories
processed_base_path = Path(__file__).parents[1] / 'text_processing' / 'processed'
processed_2020_2024_path = processed_base_path / '2020-2024'

if not processed_2020_2024_path.exists():
    print(f"Error: Directory {processed_2020_2024_path} does not exist")
else:
    print("\nChecking if these documents exist in the processed directories...")
    for celex_id in celex_ids_2020:
        dir_path = processed_2020_2024_path / celex_id
        if dir_path.exists():
            print(f"  ✓ {celex_id} exists in processed directory")
        else:
            print(f"  ✗ {celex_id} DOES NOT exist in processed directory")

# Compare with 2020 entries in searches_2020-2024.csv
searches_path = Path(__file__).parents[1] / 'text_processing' / 'searches_2020-2024.csv'
searches_df = pd.read_csv(searches_path)
celex_numbers = searches_df['CELEX number'].astype(str).tolist()

print("\nChecking if 2020 documents from random sample are in searches_2020-2024.csv...")
for celex_id in celex_ids_2020:
    # Remove the 'EU_' prefix for comparison
    celex_number = celex_id.replace('EU_', '')
    if celex_number in celex_numbers:
        print(f"  ✓ {celex_id} is present in searches_2020-2024.csv")
    else:
        print(f"  ✗ {celex_id} is NOT present in searches_2020-2024.csv")
