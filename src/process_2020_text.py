from bs4 import BeautifulSoup
import os
import codecs
import re
from pathlib import Path

def process_text(text, tp):
    """
    Process an EU law HTML file and split it into articles.
    
    Args:
        text (str): The filename of the HTML file to process
        tp (str): The time period directory name (e.g., '2020')
    """
    project_root = Path(__file__).parents[1]
    
    # Create proper paths
    input_path = project_root / "texts" / tp / text
    output_base_dir = project_root / "text_processing" / "processed" / tp
    
    # Ensure the output base directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Processing {text}...")
    
    # Read legal text
    try:
        with codecs.open(input_path, 'r', 'utf-8') as f:
            # Parse with beautiful soup
            soup = BeautifulSoup(f, 'html.parser')
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return
    except Exception as e:
        print(f"Error processing {text}: {e}")
        return
    
    # Only use body text
    body = soup.find('body')
    if not body:
        print(f"Warning: No body tag found in {text}")
        return
    
    text_only = body
    
    # Create list with paragraphs
    paragraphs = text_only.find_all('p')
    if not paragraphs:
        print(f"Warning: No paragraphs found in {text}")
        return
    
    # Structural components of legal texts
    articles_enumerated = ['Article {}'.format(i) for i in range(1, 350)]
    
    sections = ['\nSection 1\n', '\nSection 2\n', '\nSection 3\n', '\nSection 4\n', 
                '\nSection 5\n', '\nSection 6\n', '\nSection 7\n', 
                '\nSECTION 1\n', '\nSECTION 2\n', '\nSECTION 3\n', '\nSECTION 4\n', 
                '\nSECTION 5\n', '\nSECTION 6\n', '\nSection 7\n', 
                'Section 1', 'Section 2', 'Section 3', 'Section 4', 
                'Section 5', 'Section 6', 'Section 7', 
                'SECTION 1', 'SECTION 2', 'SECTION 3', 'SECTION 4', 
                'SECTION 5', 'SECTION 6', 'SECTION 7']
    
    chapters = ['CHAPTER I', 'CHAPTER II', 'CHAPTER III', 'CHAPTER IV', 'CHAPTER V', 'CHAPTER VI', 'CHAPTER VII',
                'CHAPTER 1', 'CHAPTER 2', 'CHAPTER 3', 'CHAPTER 4', 'CHAPTER 5', 'CHAPTER 6', 'CHAPTER 7',
                '\nCHAPTER I\n', '\nCHAPTER II\n', '\nCHAPTER III\n', '\nCHAPTER IV\n', '\nCHAPTER V\n', 
                '\nCHAPTER VI\n', '\nCHAPTER VII\n',
                '\nCHAPTER 1\n', '\nCHAPTER 2\n', '\nCHAPTER 3\n', '\nCHAPTER 4\n', '\nCHAPTER 5\n', 
                '\nCHAPTER 6\n', '\nCHAPTER 7\n']
    
    titles = ['TITLE I', 'TITLE II', 'TITLE III', 'TITLE IV', 'TITLE V', 'TITLE VI', 'TITLE VII', 'TITLE VIII',
             'TITLE 1', 'TITLE 2', 'TITLE 3', 'TITLE 4', 'TITLE 5', 'TITLE 6', 'TITLE 7', 'TITLE 8']
    
    i = 0  # Article counter
    j = 0  # Title counter
    k = 0  # Chapter counter
    l = 0  # Section counter
    
    # Create output directory for this specific text
    text_output_dir = output_base_dir / text[:-5]
    os.makedirs(text_output_dir, exist_ok=True)
    
    # Open new file for the front text
    file = open(text_output_dir / f"{text[:-5]}_front.txt", "w", encoding='utf-8')
    
    # Create iterable for paragraphs (useful for skipping certain paragraphs)
    paragraphs_iter = iter(paragraphs[3:])
    
    # ITERATE OVER PARAGRAPHS
    for paragraph in paragraphs_iter:
        string = paragraph.text.replace(u'\xa0', u' ')
        
        # Catch whereas
        if string == 'Whereas:':
            file.close()
            file = open(text_output_dir / f"{text[:-5]}_Whereas.txt", "w", encoding='utf-8')
        
        # Catch titles
        elif string in titles:
            j += 1
            file.close()
            file = open(text_output_dir / f"{text[:-5]}_Title_{j}_text.txt", "w", encoding='utf-8')
        
        # Catch chapters
        elif string in chapters:
            k += 1
            file.close()
            file = open(text_output_dir / f"{text[:-5]}_Title_{j}_Chapter_{k}_text.txt", "w", encoding='utf-8')
        
        # Catch sections
        elif string in sections:
            l += 1
            file.close()
            file = open(text_output_dir / f"{text[:-5]}_Title_{j}_Chapter_{k}_Section_{l}_text.txt", "w", encoding='utf-8')
                
        # Catch articles
        elif any(art in string for art in articles_enumerated):
            # If file exists, close it
            if file:
                file.close()
            
            # Get article number
            for num in range(1, 350):
                # Check if article with given number exists in the string
                if 'Article {}'.format(num) in string:
                    i = num  # Set article counter
                    break
            
            # Open new file for new article
            # This creates a different file name based on where in the legal text the article is
            if j == 0 and k == 0 and l == 0:
                # No title, chapter, section
                article_path = text_output_dir / f"{text[:-5]}_Title_0_Chapter_0_Section_0_Article_{i:02d}.txt"
            elif k == 0 and l == 0:
                # Only title
                article_path = text_output_dir / f"{text[:-5]}_Title_{j}_Chapter_0_Section_0_Article_{i:02d}.txt"
            elif l == 0:
                # Title and chapter
                article_path = text_output_dir / f"{text[:-5]}_Title_{j}_Chapter_{k}_Section_0_Article_{i:02d}.txt"
            else:
                # Title, chapter, and section
                article_path = text_output_dir / f"{text[:-5]}_Title_{j}_Chapter_{k}_Section_{l}_Article_{i:02d}.txt"
            
            file = open(article_path, "w", encoding='utf-8')
        
        # Any other text gets written to the current file
        file.write(string + '\n')
    
    # Close the final file
    if file:
        file.close()
    
    print(f"Finished processing {text}")

def main():
    """
    Process all HTML files in the texts/2020 directory
    """
    project_root = Path(__file__).parents[1]
    texts_dir = project_root / "texts" / "2020"
    
    if not texts_dir.exists():
        print(f"Error: Directory {texts_dir} does not exist")
        print("Please run the download_2020_documents.py script first")
        return
    
    texts = [f for f in os.listdir(texts_dir) if f.endswith('.html')]
    
    if not texts:
        print("No HTML files found in texts/2020/")
        return
    
    print(f"Found {len(texts)} HTML files to process")
    
    for text in texts:
        process_text(text, "2020")
    
    print("\nProcessing complete!")
    print("Next steps:")
    print("1. Run 'python src/process_2020_documents.py move' to move the processed files")
    print("   to the appropriate directory structure (text_processing/processed/2020-2024)")
    print("2. Run analyze_ner_full_dataset.py to analyze the entire dataset including 2020 documents")

if __name__ == "__main__":
    main()
