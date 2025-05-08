import requests as req
import pandas as pd
import os 

# Script to download EU laws based on CELEX numbers

# The CELEX numbers are stored in a file 'searches.csv', which for example results from 
# querying https://eur-lex.europa.eu/homepage.html for relevant policies.
# Loading the file:
exp_search_pre = 'text_processing/searches_2000-2019.csv'
exp_search_post ='text_processing/searches_2020-2024.csv'
df_pre = pd.read_csv(exp_search_pre,encoding= "ISO-8859-1")
df_post = pd.read_csv(exp_search_post,encoding= "ISO-8859-1")

# Function to take in a dataframe with CELEX numbers of EU laws and download the full text html file
# Input: Data frame with CELEX numbers of EU laws
# Output: Full text html files for each law, stored in the directory 'texts'

def get_html(df,output_path):
    os.mkdir(f'texts/{output_path}') 
    for i in range(len(df)):
        celex= df.loc[i,'CELEX number']
        url = 'https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{}&from=EN'.format(celex)
        resp = req.get(url)
        filename = '{}.html'.format(celex)
        with open( f'texts/{output_path}/' + 'EU_' + filename, "w", encoding  = 'utf-8') as file:
            file.write(resp.text,)
            file.close()

get_html(df_pre, output_path='2000-2019')
get_html(df_post, output_path='2020-2024')