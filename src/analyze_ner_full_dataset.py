import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy # For loading and using the spaCy NER model
from typing import Optional, Dict, List, Tuple
import time

# --- NER Model Setup ---
def load_spacy_ner_model(model_path: str):
    """
    Loads a trained spaCy NER model.
    """
    print(f"Attempting to load spaCy NER model from {model_path}...")
    try:
        nlp_model = spacy.load(model_path)
        print(f"Successfully loaded spaCy NER model. Entities: {nlp_model.pipe_labels.get('ner', [])}")
        return nlp_model
    except Exception as e:
        print(f"Error loading spaCy model from {model_path}: {e}")
        print("Please ensure the model path is correct and the model is trained.")
        raise

def apply_spacy_ner_model(text_content: str, nlp_model) -> list:
    """
    Applies the loaded spaCy NER model to a given text.
    Returns a list of dictionaries, where each dictionary
    represents an entity, e.g.: [{'text': 'European Union', 'label': 'ORG'}, ...]
    """
    if not nlp_model:
        raise Exception("NER model not loaded.")

    doc = nlp_model(text_content)
    extracted_entities = []
    for ent in doc.ents:
        extracted_entities.append({
            'text': ent.text,
            'label': ent.label_
        })
    return extracted_entities

# --- File System and Data Handling Functions ---
def extract_celex_id_from_path(path: Path) -> Tuple[str, int]:
    """
    Extracts the CELEX ID and year from a path.
    Assumes path contains a directory named like 'EU_32005R0184' 
    where 32005 implies year 2005.
    Returns a tuple (celex_id, year)
    """
    celex_id = path.name  # Get the directory name which is the CELEX ID
    
    # Extract year from CELEX ID format 3YYYY
    year_match = re.search(r'3(\d{4})', celex_id)
    if year_match:
        year = int(year_match.group(1))
        return celex_id, year
    
    # If we can't extract the year, return None
    return celex_id, None

def find_all_document_dirs(base_processed_path: Path) -> List[Tuple[Path, str, int]]:
    """
    Finds all document directories in the processed text data.
    Returns a list of tuples (dir_path, celex_id, year)
    """
    all_dirs = []
    
    # Process both year ranges
    for year_range in ["2000-2019", "2020-2024"]:
        year_range_dir = base_processed_path / year_range
        if not year_range_dir.is_dir():
            print(f"Warning: Year range directory not found: {year_range_dir}")
            continue
        
        # Find all subdirectories in this year range (each is a document)
        for doc_dir in year_range_dir.iterdir():
            if doc_dir.is_dir():
                celex_id, year = extract_celex_id_from_path(doc_dir)
                if year is not None:
                    all_dirs.append((doc_dir, celex_id, year))
                else:
                    print(f"Warning: Could not extract year from {doc_dir.name}, skipping.")
    
    return all_dirs

# --- Main Processing Logic ---
def process_all_documents_and_plot_trends(
    base_processed_text_path: Path,
    ner_model_path: str,
    output_csv_path: Path = None
):
    """
    Processes all text files in the processed data directory, applies NER, and plots entity trends.
    Optionally saves the extracted entities to a CSV file.
    """
    start_time = time.time()
    nlp_ner_model = load_spacy_ner_model(ner_model_path)
    
    # Find all document directories
    print("Finding all document directories...")
    all_doc_dirs = find_all_document_dirs(base_processed_text_path)
    print(f"Found {len(all_doc_dirs)} document directories.")
    
    all_extracted_entities = []
    processed_docs_count = 0
    total_text_files_processed = 0
    
    print(f"\nProcessing documents...")
    
    for doc_dir, celex_id, doc_year in all_doc_dirs:
        # Find all .txt files in this document directory
        text_files = list(doc_dir.rglob('*.txt'))
        
        if not text_files:
            continue
        
        processed_docs_count += 1
        for txt_file_path in text_files:
            total_text_files_processed += 1
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {txt_file_path} for {celex_id}: {e}, skipping.")
                continue
            
            if not content.strip():  # Skip empty files
                continue
            
            entities_from_file = apply_spacy_ner_model(content, nlp_ner_model)
            
            for entity in entities_from_file:
                all_extracted_entities.append({
                    'year': doc_year,
                    'entity_label': entity['label'],
                    'entity_text': entity['text'],
                    'celex_id': celex_id
                })
        
        # Print progress every 50 documents
        if processed_docs_count % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"  Processed {processed_docs_count}/{len(all_doc_dirs)} documents... ({elapsed_time:.2f} seconds elapsed)")
    
    print(f"\nFinished processing documents.")
    print(f"Total documents processed: {processed_docs_count}")
    print(f"Total individual .txt files processed: {total_text_files_processed}")
    
    if not all_extracted_entities:
        print("No entities were extracted. Cannot generate plot.")
        return
    
    print(f"Total entities extracted: {len(all_extracted_entities)}")
    
    # Create DataFrame from extracted entities
    results_df = pd.DataFrame(all_extracted_entities)
    
    # Save to CSV if path provided
    if output_csv_path:
        results_df.to_csv(output_csv_path, index=False)
        print(f"Extracted entities saved to {output_csv_path}")
    
    if results_df.empty:
        print("DataFrame of results is empty. No data to plot.")
        return
    
    # Aggregate data: count entities per label per year
    aggregated_df = results_df.groupby(['year', 'entity_label']).size().reset_index(name='count')
    
    # Pivot for plotting: years as index, entity_labels as columns
    pivot_table_df = aggregated_df.pivot(index='year', columns='entity_label', values='count').fillna(0)
    
    if pivot_table_df.empty:
        print("Pivot table is empty. Cannot plot.")
        return
    
    # Ensure all expected entity types from the model are columns, even if some have zero counts
    expected_labels = nlp_ner_model.pipe_labels.get('ner', [])
    for label in expected_labels:
        if label not in pivot_table_df.columns:
            pivot_table_df[label] = 0
    
    pivot_table_df = pivot_table_df.sort_index()  # Sort years
    
    # Save aggregated data to CSV
    aggregated_csv_path = Path("data/entity_counts_by_year_type.csv")
    aggregated_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated entity counts by year and type saved to {aggregated_csv_path}")
    
    # Plotting
    plt.figure(figsize=(18, 10))
    for entity_label_column in pivot_table_df.columns:
        if pivot_table_df[entity_label_column].sum() > 0:  # Only plot if there's data
            plt.plot(pivot_table_df.index.astype(str), pivot_table_df[entity_label_column], marker='o', linestyle='-', label=entity_label_column)
    
    plt.title('NER Entity Counts Over Time (Full Dataset)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Entities', fontsize=14)
    
    if pivot_table_df.empty or all(pivot_table_df[col].sum() == 0 for col in pivot_table_df.columns):
        plt.text(0.5, 0.5, "No entities found to plot.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    else:
        plt.legend(title='Entity Types', bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for legend
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    output_plot_file = "ner_entity_trends_full_dataset.png"
    try:
        plt.savefig(output_plot_file)
        print(f"Plot successfully saved to {output_plot_file}")
        # plt.show()  # Uncomment to display plot if running in an interactive environment
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == '__main__':
    # Assuming the script is run from the project's root directory.
    project_root = Path(".")
    
    # Base path to your processed text data (where 2000-2019, 2020-2024 folders are)
    path_to_base_processed_data = project_root / 'text_processing' / 'processed'
    
    # Path to save the CSV with all extracted entities (optional)
    output_csv_path = project_root / 'data' / 'extracted_entities_full_dataset.csv'
    
    # --- IMPORTANT ---
    # Path to your trained spaCy NER model directory.
    # This is the 'model-best' directory created by `spacy train`.
    your_spacy_ner_model_path = "notebooks/ner_data/model/model-best"
    
    if not (project_root / "notebooks" / "ner_data" / "model" / "model-best").exists():
        print("*" * 50)
        print(f"WARNING: NER model directory not found at {your_spacy_ner_model_path}")
        print("Please ensure the path is correct and the model exists.")
        print("You might need to train it first using the 'traditional_nlp_pipeline_update.ipynb' notebook.")
        print("*" * 50)
    else:
        process_all_documents_and_plot_trends(
            base_processed_text_path=path_to_base_processed_data,
            ner_model_path=your_spacy_ner_model_path,
            output_csv_path=output_csv_path
        )
