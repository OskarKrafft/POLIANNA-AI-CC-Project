#!/usr/bin/env python3
# filepath: /Users/oskarkrafft/Desktop/Projects/POLIANNA-AI-CC-Project/src/visualize_ner_with_fit_for_55_vertical.py
"""
Visualization script for NER data with vertical dashed red line for 'Fit for 55' Policy Package.
Creates a figure with two vertically stacked subfigures:
1. Total NER entity counts by year
2. Normalized NER entity counts by year
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

def create_ner_visualization_with_fit_for_55_vertical():
    """
    Creates a visualization of NER data with a vertical dashed red line for 
    the 'Fit for 55' Policy Package in 2021.
    Shows both total counts and normalized counts as vertically stacked subfigures.
    """
    project_root = Path(".")
    entity_counts_path = project_root / 'data' / 'entity_counts_by_year_type.csv'
    
    # Check if the entity counts file exists
    if not entity_counts_path.exists():
        print(f"Error: Entity counts file not found at {entity_counts_path}")
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
    
    # Create pivot tables for plotting
    pivot_total = entity_counts_df.pivot(index='year', columns='entity_label', values='count').fillna(0)
    pivot_total = pivot_total.sort_index()
    
    pivot_normalized = merged_df.pivot(index='year', columns='entity_label', values='normalized_count').fillna(0)
    pivot_normalized = pivot_normalized.sort_index()
    
    # Create the visualization with two vertically stacked subfigures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 16))
    
    # Color mapping for consistent entity colors across plots
    entity_colors = {
        'ADDRESSEE': '#1f77b4',  # blue
        'AUTHORITY': '#ff7f0e',  # orange
        'SECTOR': '#2ca02c',     # green
    }
    
    # Plot 1: Total entity counts
    for entity_type in pivot_total.columns:
        color = entity_colors.get(entity_type, None)
        ax1.plot(pivot_total.index.astype(str), pivot_total[entity_type], 
                marker='o', linestyle='-', label=entity_type, color=color, linewidth=2)
    
    ax1.set_title('Total NER Entity Counts Over Time', fontsize=16)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Number of Entities', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add 'Fit for 55' vertical line for 2021
    ax1.axvline(x='2021', color='red', linestyle='--', linewidth=2)
    
    # Plot 2: Normalized entity counts
    for entity_type in pivot_normalized.columns:
        color = entity_colors.get(entity_type, None)
        ax2.plot(pivot_normalized.index.astype(str), pivot_normalized[entity_type], 
                marker='o', linestyle='-', label=entity_type, color=color, linewidth=2)
    
    ax2.set_title('Normalized NER Entity Counts Over Time', fontsize=16)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel('Entities per Document', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add 'Fit for 55' vertical line for 2021
    ax2.axvline(x='2021', color='red', linestyle='--', linewidth=2)
    
    # Add a single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title='Entity Types', 
               loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
    
    # Add a title for the entire figure
    fig.suptitle('NER Entity Trends (2021: Fit for 55 Package)', fontsize=20)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.97])
    
    # Save the plot
    output_path = project_root / 'ner_entity_trends_fit_for_55_report_vertical.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVertical plot saved to {output_path}")

if __name__ == '__main__':
    create_ner_visualization_with_fit_for_55_vertical()
