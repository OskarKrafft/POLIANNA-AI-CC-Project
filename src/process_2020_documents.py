import os
import shutil
from pathlib import Path
import sys

def setup_for_processing():
    """
    Prepare the environment for processing 2020 documents.
    1. Create a text_processing/texts/2020 directory if it doesn't exist
    2. Create a text_processing/processed/2020 directory if it doesn't exist
    """
    project_root = Path(__file__).parents[1]
    
    # Check if the downloaded HTML files exist
    texts_2020_dir = project_root / 'texts' / '2020'
    if not texts_2020_dir.exists() or not any(texts_2020_dir.glob('*.html')):
        print("Error: No HTML files found in texts/2020/")
        print("Please run the download_2020_documents.py script first.")
        return False
    
    # Create the processed/2020 directory if it doesn't exist
    processed_2020_dir = project_root / 'text_processing' / 'processed' / '2020'
    if not processed_2020_dir.exists():
        os.makedirs(processed_2020_dir)
        print(f"Created directory: {processed_2020_dir}")
    
    print("\nSetup complete. Now you should:")
    print("1. Open and run the process_text.ipynb notebook")
    print("2. Make sure to process the texts/2020/ directory")
    print("3. After processing, run this script again with the 'move' parameter to move the files")
    print("   Example: python src/process_2020_documents.py move")
    
    return True

def move_processed_files():
    """
    Move processed files from text_processing/processed/2020 to text_processing/processed/2020-2024
    """
    project_root = Path(__file__).parents[1]
    processed_2020_dir = project_root / 'text_processing' / 'processed' / '2020'
    target_dir = project_root / 'text_processing' / 'processed' / '2020-2024'
    
    if not processed_2020_dir.exists() or not any(processed_2020_dir.iterdir()):
        print("Error: No processed files found in text_processing/processed/2020/")
        print("Please process the 2020 documents first using process_text.ipynb.")
        return False
    
    if not target_dir.exists():
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    # Move all subdirectories from processed/2020 to processed/2020-2024
    moved_count = 0
    for item in processed_2020_dir.iterdir():
        if item.is_dir():
            target_path = target_dir / item.name
            if target_path.exists():
                print(f"Warning: {target_path} already exists, skipping...")
                continue
                
            shutil.move(str(item), str(target_dir))
            moved_count += 1
            print(f"Moved {item.name} to {target_dir}")
    
    print(f"\nMoved {moved_count} directories to {target_dir}")
    print("\nNext steps:")
    print("1. Run the analyze_ner_full_dataset.py script to analyze the entire dataset including 2020 documents")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "move":
        move_processed_files()
    else:
        setup_for_processing()
