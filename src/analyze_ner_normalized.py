import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def create_normalized_entity_plot():
    """
    Creates a normalized entity plot that shows entity counts divided by 
    the number of documents published each year.
    """
    # Load the aggregated entity counts
    project_root = Path(".")
    entity_counts_path = project_root / 'data' / 'entity_counts_by_year_type.csv'
    
    if not entity_counts_path.exists():
        print(f"Error: Entity counts file not found at {entity_counts_path}")
        print("Please run analyze_ner_full_dataset.py first to generate this file.")
        return

    # Load the full dataset extraction file to get document counts
    full_dataset_path = project_root / 'data' / 'extracted_entities_full_dataset.csv'
    
    if not full_dataset_path.exists():
        print(f"Error: Full dataset file not found at {full_dataset_path}")
        print("Please run analyze_ner_full_dataset.py first to generate this file.")
        return
    
    # Load the data
    print("Loading entity data...")
    entity_counts_df = pd.read_csv(entity_counts_path)
    entities_df = pd.read_csv(full_dataset_path)
    
    # Get counts of unique documents per year
    print("Calculating document counts per year...")
    docs_per_year = entities_df.groupby('year')['celex_id'].nunique().reset_index()
    docs_per_year.columns = ['year', 'document_count']
    
    # Print document counts for verification
    print("\nNumber of documents per year:")
    for _, row in docs_per_year.sort_values('year').iterrows():
        print(f"  {int(row['year'])}: {row['document_count']} documents")
    
    # Merge the entity counts with document counts
    merged_df = pd.merge(entity_counts_df, docs_per_year, on='year')
    
    # Calculate normalized entity counts (entities per document)
    merged_df['normalized_count'] = merged_df['count'] / merged_df['document_count']
    
    # Create a pivot table for plotting
    pivot_df = merged_df.pivot(index='year', columns='entity_label', values='normalized_count').fillna(0)
    
    # Sort years for plotting
    pivot_df = pivot_df.sort_index()
    
    # Plotting
    plt.figure(figsize=(18, 10))
    
    # Plot the normalized entity counts
    for entity_type in pivot_df.columns:
        plt.plot(pivot_df.index.astype(str), pivot_df[entity_type], 
                 marker='o', linestyle='-', label=entity_type)
    
    plt.title('Normalized NER Entity Counts Over Time (Full Dataset)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Entities per Document', fontsize=14)
    plt.legend(title='Entity Types', bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False))
    
    # Save the plot
    output_path = project_root / 'ner_entity_trends_normalized.png'
    plt.savefig(output_path)
    print(f"\nNormalized plot saved to {output_path}")

    # Create a separate plot for raw document counts over time
    plt.figure(figsize=(14, 8))
    docs_per_year_sorted = docs_per_year.sort_values('year')
    plt.bar(docs_per_year_sorted['year'].astype(str), docs_per_year_sorted['document_count'], color='steelblue')
    plt.title('Number of Documents Analyzed by Year', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Document Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the document count plot
    doc_count_output_path = project_root / 'document_counts_by_year.png'
    plt.savefig(doc_count_output_path)
    print(f"Document count plot saved to {doc_count_output_path}")

if __name__ == '__main__':
    create_normalized_entity_plot()
