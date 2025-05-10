"""
Utility functions for policy text annotation system.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from collections import Counter


def get_project_root() -> Path:
    """
    Find and return the project root directory.
    Looks for the 'data' directory as a marker for the project root.
    
    Returns:
        Path: Project root directory
    """
    try:
        # First try to use a project-wide definition if available
        from definitions import ROOT_DIR
        return Path(ROOT_DIR)
    except ImportError:
        # Fallback to finding the root directory based on current location
        current_dir = Path(os.getcwd())
        # Look for the 'data' directory going up the tree
        root_dir = current_dir
        while root_dir.name and not (root_dir / 'data').exists():
            root_dir = root_dir.parent
        
        if not (root_dir / 'data').exists():
            raise FileNotFoundError(
                "Could not find project root directory containing 'data' folder. "
                "Please run from within the project or set ROOT_DIR in definitions.py."
            )
        
        return root_dir


def load_coding_scheme(filepath: Optional[str] = None, scheme_name: Optional[str] = None) -> Dict:
    """
    Load the coding scheme from the JSON file.
    
    Args:
        filepath: Optional custom path to the coding scheme JSON file.
                 If None, uses the default path.
        scheme_name: Optional name of a specific coding scheme file to use.
                     If provided, overrides filepath and loads "{scheme_name}.json"
                     from the standard location.
    
    Returns:
        Dict: The coding scheme as a dictionary
    """
    if scheme_name is not None:
        root_dir = get_project_root()
        filepath = os.path.join(root_dir, 'data', '01_policy_info', f"{scheme_name}.json")
    elif filepath is None:
        root_dir = get_project_root()
        filepath = os.path.join(root_dir, 'data', '01_policy_info', 'Coding_Scheme.json')
    
    try:
        with open(filepath, 'r') as f:
            coding_scheme = json.load(f)
        return coding_scheme
    except FileNotFoundError:
        raise FileNotFoundError(f"Coding scheme file not found at {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in coding scheme file at {filepath}")


def filter_coding_scheme(coding_scheme: Dict, layers: Optional[List[str]] = None, 
                         tagsets: Optional[List[str]] = None) -> Dict:
    """
    Filter coding scheme to only include selected layers/tagsets.
    
    Args:
        coding_scheme: Full coding scheme dictionary
        layers: List of layer names to include (None for all)
        tagsets: List of tagset names to include (None for all)
    
    Returns:
        Dict: Filtered coding scheme
    """
    if not layers and not tagsets:
        return coding_scheme
    
    if "layers" not in coding_scheme:
        raise ValueError("Coding scheme JSON does not contain a 'layers' key")
    
    filtered = {"layers": []}
    
    for layer in coding_scheme.get("layers", []):
        if "layer" not in layer:
            raise KeyError(f"Layer object missing 'layer' field: {layer}")
        
        if layers and layer["layer"] not in layers:
            continue
        
        new_layer = {k: v for k, v in layer.items() if k != "tagsets"}
        new_layer["tagsets"] = []
        
        for tagset in layer.get("tagsets", []):
            if "tagset" not in tagset:
                raise KeyError(f"Tagset object missing 'tagset' field: {tagset}")
            
            if tagsets and tagset["tagset"] not in tagsets:
                continue
            
            new_layer["tagsets"].append(tagset)
        
        if new_layer["tagsets"] or not tagsets:
            filtered["layers"].append(new_layer)
    
    if not filtered["layers"]:
        raise ValueError(f"No layers/tagsets matched your filter. layers={layers}, tagsets={tagsets}")
    
    return filtered


def create_extended_coding_scheme(
    output_name: str = "Coding_Scheme_Extended",
    min_occurrences: int = 2,
    max_examples: int = 5
) -> str:
    """
    Create an extended coding scheme with examples extracted from annotated data.
    
    Args:
        output_name: Name of the output file (without extension)
        min_occurrences: Minimum number of occurrences for an example to be included
        max_examples: Maximum number of examples to include per tag
    
    Returns:
        str: Path to the created extended coding scheme file
    """
    # Load the original coding scheme
    original_scheme = load_coding_scheme()
    
    # Get all annotated articles
    articles = get_articles_paths()
    
    # Create a dictionary to collect examples for each tag
    tag_examples = {}
    
    # Process all annotated articles to collect examples
    for article in articles:
        try:
            # Load the curated annotations
            annotations = load_curated_annotations(article['id'])
            
            # Process each annotation
            for annotation in annotations:
                layer = annotation.get('layer')
                feature = annotation.get('feature')  # feature is the tagset
                tag = annotation.get('tag')
                text = annotation.get('text')
                
                if not all([layer, feature, tag, text]):
                    continue
                
                # Create a unique identifier for the tag
                tag_id = f"{layer}|{feature}|{tag}"
                
                # Add the example text to the collection
                if tag_id not in tag_examples:
                    tag_examples[tag_id] = []
                
                tag_examples[tag_id].append(text)
                
        except Exception as e:
            print(f"Error processing article {article['id']}: {e}")
    
    # Create a copy of the original scheme to modify
    extended_scheme = json.loads(json.dumps(original_scheme))
    
    # Count occurrences and filter based on min_occurrences
    tag_examples_filtered = {}
    for tag_id, examples in tag_examples.items():
        # Count occurrences of each example
        example_counts = Counter(examples)
        
        # Filter by minimum occurrences
        filtered_examples = [(ex, count) for ex, count in example_counts.items() 
                          if count >= min_occurrences]
        
        # Sort by frequency (descending)
        filtered_examples.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max_examples
        filtered_examples = filtered_examples[:max_examples]
        
        # Store only the example text (not the count)
        tag_examples_filtered[tag_id] = [ex for ex, _ in filtered_examples]
    
    # Add the examples to the extended scheme
    for layer in extended_scheme.get('layers', []):
        layer_name = layer.get('layer')
        
        for tagset in layer.get('tagsets', []):
            tagset_name = tagset.get('tagset')
            
            for tag in tagset.get('tags', []):
                tag_name = tag.get('tag_name')
                
                # Create the tag ID
                tag_id = f"{layer_name}|{tagset_name}|{tag_name}"
                
                # Add examples if available
                if tag_id in tag_examples_filtered and tag_examples_filtered[tag_id]:
                    tag['tag_examples'] = tag_examples_filtered[tag_id]
                else:
                    tag['tag_examples'] = []
    
    # Save the extended scheme
    root_dir = get_project_root()
    output_path = os.path.join(root_dir, 'data', '01_policy_info', f"{output_name}.json")
    
    with open(output_path, 'w') as f:
        json.dump(extended_scheme, f, indent=2)
    
    # Print statistics
    total_tags = sum(len(tagset.get('tags', [])) 
                     for layer in extended_scheme.get('layers', []) 
                     for tagset in layer.get('tagsets', []))
    
    tags_with_examples = sum(1 for tag_id in tag_examples_filtered if tag_examples_filtered[tag_id])
    
    print(f"Extended coding scheme created at {output_path}")
    print(f"Added examples to {tags_with_examples} out of {total_tags} tags")
    print(f"Minimum occurrences: {min_occurrences}, Maximum examples per tag: {max_examples}")
    
    return output_path


def get_articles_paths(directory: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Get paths to all articles with raw text and curated annotations.
    
    Args:
        directory: Optional custom path to the articles directory.
                  If None, uses the default path.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries with article info including paths
    """
    if directory is None:
        root_dir = get_project_root()
        directory = os.path.join(root_dir, 'data', '03b_processed_to_json')
    
    article_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    articles = []
    for article_dir in article_dirs:
        raw_text_path = os.path.join(directory, article_dir, 'Raw_Text.txt')
        annotations_path = os.path.join(directory, article_dir, 'Curated_Annotations.json')
        
        if os.path.exists(raw_text_path) and os.path.exists(annotations_path):
            articles.append({
                'id': article_dir,
                'raw_text_path': raw_text_path,
                'annotations_path': annotations_path
            })
    
    return articles


def load_raw_text(article_id: str) -> str:
    """
    Load the raw text of a specific article.
    
    Args:
        article_id: Article identifier (folder name in 03b_processed_to_json)
    
    Returns:
        str: Raw text of the article
    """
    root_dir = get_project_root()
    filepath = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id, 'Raw_Text.txt')
    
    try:
        with open(filepath, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw text file not found for article {article_id}")


def load_curated_annotations(article_id: str) -> List[Dict]:
    """
    Load the curated annotations for a specific article.
    
    Args:
        article_id: Article identifier (folder name in 03b_processed_to_json)
    
    Returns:
        List[Dict]: Curated annotations for the article
    """
    root_dir = get_project_root()
    filepath = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id, 'Curated_Annotations.json')
    
    try:
        with open(filepath, 'r') as f:
            annotations = json.load(f)
        return annotations
    except FileNotFoundError:
        raise FileNotFoundError(f"Curated annotations file not found for article {article_id}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in curated annotations file for article {article_id}")


def load_coder_annotations(article_id: str, coder_id: Optional[str] = None) -> Union[Dict[str, List[Dict]], List[Dict]]:
    """
    Load the coder annotations for a specific article.
    
    Args:
        article_id: Article identifier (folder name in 03b_processed_to_json)
        coder_id: Optional identifier of the specific coder (e.g., "A", "B", etc.)
                 If None, returns all coders' annotations.
    
    Returns:
        Union[Dict[str, List[Dict]], List[Dict]]: Coder annotations for the article.
        If coder_id is None, returns a dictionary mapping coder IDs to their annotations.
        If coder_id is specified, returns just that coder's annotations as a list.
    """
    root_dir = get_project_root()
    filepath = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id, 'Coder_Annotations.json')
    
    try:
        with open(filepath, 'r') as f:
            all_annotations = json.load(f)
        
        if coder_id is not None:
            if coder_id in all_annotations:
                return all_annotations[coder_id]
            else:
                raise ValueError(f"Coder ID '{coder_id}' not found in annotations for article {article_id}")
        
        return all_annotations
    except FileNotFoundError:
        raise FileNotFoundError(f"Coder annotations file not found for article {article_id}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in coder annotations file for article {article_id}")


def save_generated_annotations(article_id: str, annotations: List[Dict]) -> str:
    """
    Save generated annotations to a JSON file.
    
    Args:
        article_id: Article identifier (folder name in 03b_processed_to_json)
        annotations: List of annotation dictionaries
    
    Returns:
        str: Path to the saved file
    """
    root_dir = get_project_root()
    directory = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id)
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found for article {article_id}")
    
    filepath = os.path.join(directory, 'Generated_Annotations.json')
    
    with open(filepath, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return filepath


def filter_annotations(annotations: List[Dict], layers: Optional[List[str]] = None,
                      tagsets: Optional[List[str]] = None) -> List[Dict]:
    """
    Filter annotations to only include specific layers/tagsets.
    
    Args:
        annotations: List of annotation dictionaries
        layers: List of layer names to include (None for all)
        tagsets: List of tagset names to include (None for all)
    
    Returns:
        List[Dict]: Filtered annotations
    """
    if not layers and not tagsets:
        return annotations
    
    filtered = []
    for annotation in annotations:
        if layers and annotation.get('layer') not in layers:
            continue
        
        if tagsets and annotation.get('feature') not in tagsets:
            continue
        
        filtered.append(annotation)
    
    return filtered 