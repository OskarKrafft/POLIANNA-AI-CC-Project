import requests as req
import pandas as pd
import os
from pathlib import Path

# Script to download 2020 EU laws based on CELEX numbers

# Loading the file:
project_root = Path(__file__).parents[1]
exp_search_2020 = project_root / 'text_processing' / 'searches_2020.csv'
df_2020 = pd.read_csv(exp_search_2020, encoding="ISO-8859-1")

# Function to take in a dataframe with CELEX numbers of EU laws and download the full text html file
# Input: Data frame with CELEX numbers of EU laws
# Output: Full text html files for each law, stored in the directory 'texts/2020'

def get_html(df, output_path):
    texts_dir = project_root / 'texts'
    output_dir = texts_dir / output_path
    
    # Create texts directory if it doesn't exist
    if not texts_dir.exists():
        os.mkdir(texts_dir)
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        os.mkdir(output_dir)
    
    print(f"Downloading {len(df)} CELEX documents for {output_path}...")
    
    for i in range(len(df)):
        celex = df.loc[i, 'CELEX number']
        url = f'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}&from=EN'
        print(f"Downloading {celex} [{i+1}/{len(df)}]")
        
        try:
            resp = req.get(url)
            resp.raise_for_status()  # Raise an exception for HTTP errors
            
            filename = f'EU_{celex}.html'
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding='utf-8') as file:
                file.write(resp.text)
                print(f"Successfully downloaded {filename}")
                
        except req.RequestException as e:
            print(f"Error downloading {celex}: {e}")
            continue

if __name__ == "__main__":
    get_html(df_2020, output_path='2020')
    print("\nNext steps:")
    print("1. Run the process_text.ipynb notebook to process the downloaded HTML files")
    print("2. Move the processed files to the appropriate directory structure")
    print("   (from text_processing/processed/2020 to text_processing/processed/2020-2024)")
