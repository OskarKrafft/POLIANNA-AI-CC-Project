"""
Annotation module for LLM-based policy text annotation.
"""

import os
import json
import random
import re
from typing import Dict, List, Optional, Any, Union, Tuple
import requests
from pathlib import Path

# Import utility functions
from src.agent.utils import (
    get_project_root, 
    load_coding_scheme, 
    filter_coding_scheme,
    load_raw_text,
    load_curated_annotations,
    save_generated_annotations
)


def get_anthropic_api_key() -> str:
    """
    Get the Anthropic API key from environment variables.
    
    Returns:
        str: Anthropic API key
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please set it in .env file or export it directly."
        )
    return api_key


def get_model_parameters() -> Dict[str, Any]:
    """
    Get LLM model parameters from environment variables.
    
    Returns:
        Dict[str, Any]: Model parameters
    """
    return {
        'model': os.getenv('LLM_MODEL', 'claude-3-opus-20240229'),
        'temperature': float(os.getenv('LLM_TEMPERATURE', '0.2')),
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '4000')),
    }


def prepare_annotation_prompt(
    raw_text: str, 
    coding_scheme: Dict,
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    few_shot_examples: Optional[List[Dict]] = None,
    use_extended_scheme: bool = True
) -> str:
    """
    Prepare the prompt for the LLM to annotate policy text.
    
    Args:
        raw_text: The policy text to annotate
        coding_scheme: The coding scheme dictionary (potentially filtered)
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        few_shot_examples: Optional list of few-shot examples
        use_extended_scheme: Whether to use the extended scheme with examples
    
    Returns:
        str: The formatted prompt for the LLM
    """
    # Filter coding scheme if necessary
    if layers or tagsets:
        filtered_scheme = filter_coding_scheme(coding_scheme, layers, tagsets)
    else:
        filtered_scheme = coding_scheme
    
    # Create the base prompt
    prompt = "# Policy Text Annotation Task\n\n"
    prompt += "You are an expert policy analyst helping to annotate policy text with a specific coding scheme.\n\n"
    
    # Add information about selective annotation if applicable
    if layers or tagsets:
        prompt += "## Annotation Focus\n\n"
        if layers:
            prompt += f"Focus on these layers only: {', '.join(layers)}\n"
        if tagsets:
            prompt += f"Focus on these tagsets only: {', '.join(tagsets)}\n"
        prompt += "\n"
    
    # Add the coding scheme
    prompt += "## Coding Scheme\n\n"
    
    # Check if we need to transform the scheme to make examples more accessible
    if use_extended_scheme and any('tag_examples' in tag for layer in filtered_scheme.get('layers', []) 
                                for tagset in layer.get('tagsets', []) 
                                for tag in tagset.get('tags', [])):
        # Adjust the scheme to present examples nicely
        prompt += "The coding scheme includes descriptions and examples for each tag:\n\n"
        
        for layer in filtered_scheme.get('layers', []):
            layer_name = layer.get('layer')
            prompt += f"### Layer: {layer_name}\n\n"
            
            if layer.get('layer_description'):
                prompt += f"{layer.get('layer_description')}\n\n"
            
            for tagset in layer.get('tagsets', []):
                tagset_name = tagset.get('tagset')
                prompt += f"#### Tagset: {tagset_name}\n\n"
                
                if tagset.get('tagset_description'):
                    prompt += f"{tagset.get('tagset_description')}\n\n"
                
                for tag in tagset.get('tags', []):
                    tag_name = tag.get('tag_name')
                    tag_description = tag.get('tag_description', '')
                    tag_examples = tag.get('tag_examples', [])
                    
                    prompt += f"##### Tag: {tag_name}\n\n"
                    prompt += f"Description: {tag_description}\n\n"
                    
                    if tag_examples:
                        prompt += "Examples:\n"
                        for example in tag_examples:
                            prompt += f"- \"{example}\"\n"
                        prompt += "\n"
                    else:
                        prompt += "No examples available for this tag.\n\n"
            
            prompt += "\n"
    else:
        # Use the standard JSON representation
        prompt += json.dumps(filtered_scheme, indent=2)
        prompt += "\n\n"
    
    # Add few-shot examples if provided
    if few_shot_examples and len(few_shot_examples) > 0:
        prompt += "## Examples\n\n"
        for i, example in enumerate(few_shot_examples):
            prompt += f"### Example {i+1}\n\n"
            prompt += f"Text: {example['text']}\n\n"
            
            # Show simplified examples without indices
            simplified_anns = []
            for ann in example['annotations']:
                simplified_anns.append({
                    'layer': ann.get('layer'),
                    'feature': ann.get('feature'),
                    'tag': ann.get('tag'),
                    'text': ann.get('text')
                })
            
            prompt += f"Annotations: {json.dumps(simplified_anns, indent=2)}\n\n"
    
    # Add the text to annotate
    prompt += "## Text to Annotate\n\n"
    prompt += raw_text
    prompt += "\n\n"
    
    # Add improved instructions
    prompt += "## Instructions\n\n"
    prompt += "1. Identify spans in the text that should be annotated according to the coding scheme.\n"
    prompt += "2. For each span, determine the appropriate layer, feature, and tag from the coding scheme.\n"
    prompt += "3. Output a JSON array where each object represents an annotation with these fields:\n"
    prompt += "   - layer: The layer name from the coding scheme\n"
    prompt += "   - feature: The feature/tagset name\n"
    prompt += "   - tag: The specific tag name\n"
    prompt += "   - text: The exact text of the span\n\n"
    
    # IMPORTANT NOTE FOR THE LLM
    prompt += "### Important Note\n\n"
    #prompt += "DO NOT include character indexes (start/stop) in your response. Focus only on identifying the correct spans and their tags.\n"
    prompt += "Make sure to extract the complete, exact phrases from the text that represent each entity - don't paraphrase or shorten them.\n\n"
    
    # Add specific guidance about overlapping entities
    prompt += "### Handling Multiple Roles\n\n"
    prompt += "Pay careful attention to entities that may have multiple roles or tags. For example, 'member states' might simultaneously be:\n"
    prompt += "- Addressee_default (being addressed directly)\n"
    prompt += "- Authority_monitoring (having monitoring responsibilities)\n"
    prompt += "- Addressee_monitored (being monitored by others)\n\n"
    prompt += "When you encounter such entities, create separate annotations for EACH role they play in the context, but do not h.\n\n"
    
    # Add a note about the annotation approach
    prompt += "### Annotation Approach\n\n"
    prompt += "Follow these principles when annotating:\n"
    prompt += "1. Annotate at the span level (one or more tokens).\n"
    prompt += "2. Identify both the boundaries of relevant spans and their correct labels.\n"
    prompt += "3. Annotate the shortest acceptable span to be meaningful. Don't include unnecessary articles like 'the' or 'a'.\n"
    prompt += "4. Recognize that spans can overlap and entities can have multiple tags.\n"
    prompt += "5. Consider the context to determine the appropriate tag for each entity.\n\n"
    
    # Add note about examples
    if use_extended_scheme:
        prompt += "### Note on Examples\n\n"
        prompt += "The examples provided for each tag are not exhaustive. You can and should annotate other relevant spans "\
                 "that match the tag description, even if they are not similar to the provided examples.\n\n"
    
    prompt += "Output only the JSON array with no additional explanation or text."
    
    return prompt


def load_few_shot_examples(
    num_examples: int = 3,
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    max_text_length: int = 500,
    exclude_article_ids: Optional[List[str]] = None,
    use_extended_scheme: bool = True
) -> List[Dict]:
    """
    Load few-shot examples from curated annotations.
    
    Args:
        num_examples: Number of examples to load
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        max_text_length: Maximum length of text for examples
        exclude_article_ids: Article IDs to exclude from examples
        use_extended_scheme: Whether to use the extended scheme examples
    
    Returns:
        List[Dict]: Few-shot examples
    """
    root_dir = get_project_root()
    examples_dir = os.path.join(root_dir, 'data', '03b_processed_to_json')
    
    # Get all article directories
    article_dirs = [d for d in os.listdir(examples_dir) 
                   if os.path.isdir(os.path.join(examples_dir, d))
                   and (not exclude_article_ids or d not in exclude_article_ids)]
    
    random.shuffle(article_dirs)
    
    examples = []
    for article_dir in article_dirs[:num_examples * 2]:  # Get more than needed in case some fail
        try:
            # Load raw text
            with open(os.path.join(examples_dir, article_dir, 'Raw_Text.txt'), 'r') as f:
                text = f.read().strip()
            
            # Skip if text is too long
            if len(text) > max_text_length:
                continue
            
            # Load annotations
            with open(os.path.join(examples_dir, article_dir, 'Curated_Annotations.json'), 'r') as f:
                annotations = json.load(f)
            
            # Filter annotations if necessary
            if layers or tagsets:
                filtered_annotations = []
                for ann in annotations:
                    if layers and ann.get('layer') not in layers:
                        continue
                    if tagsets and ann.get('feature') not in tagsets:
                        continue
                    # Remove span_id and tokens
                    if 'span_id' in ann:
                        del ann['span_id']
                    if 'tokens' in ann:
                        del ann['tokens']
                    filtered_annotations.append(ann)
                
                # Skip if no annotations remain after filtering
                if not filtered_annotations:
                    continue
                
                annotations = filtered_annotations
            else:
                # Remove span_id and tokens from all annotations
                for ann in annotations:
                    if 'span_id' in ann:
                        del ann['span_id']
                    if 'tokens' in ann:
                        del ann['tokens']
            
            examples.append({
                'text': text,
                'annotations': annotations
            })
            
            # Break if we have enough examples
            if len(examples) >= num_examples:
                break
                
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    
    return examples[:num_examples]


def call_anthropic_api(prompt: str) -> str:
    """
    Call the Anthropic API with the given prompt.
    
    Args:
        prompt: The prompt to send to the API
    
    Returns:
        str: The API response content
    """
    api_key = get_anthropic_api_key()
    model_params = get_model_parameters()
    
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
    }
    
    data = {
        'model': model_params['model'],
        'max_tokens': model_params['max_tokens'],
        'temperature': model_params['temperature'],
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }
    
    response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)
    response.raise_for_status()
    content = response.json()['content'][0]['text']
    
    return content


def parse_annotations_from_response(response: str) -> List[Dict]:
    """
    Parse annotations from the LLM response.
    
    Args:
        response: The LLM response text
    
    Returns:
        List[Dict]: List of annotation dictionaries
    """
    # Extract JSON content from the response
    # This handles the case where the LLM might add explanatory text
    try:
        # Try to parse the entire response as JSON
        annotations = json.loads(response)
        return annotations
    except json.JSONDecodeError:
        # If that fails, try to extract JSON array from text
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                annotations = json.loads(json_str)
                return annotations
        except (json.JSONDecodeError, ValueError):
            raise ValueError(f"Failed to parse annotations from LLM response:\n{response}")


def find_spans_in_text(text: str, span_texts: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all occurrences of each span text in the full text.
    
    Args:
        text: The full text to search in
        span_texts: List of span texts to find
    
    Returns:
        Dict[str, List[Tuple[int, int]]]: Dictionary mapping each span text to a list of (start, stop) positions
    """
    span_positions = {}
    
    for span_text in span_texts:
        # Skip empty spans
        if not span_text or span_text.isspace():
            continue
            
        # Clean the span text by trimming excess whitespace
        clean_span = span_text.strip()
        
        # Escape special regex characters
        escaped_text = re.escape(clean_span)
        
        # Find all occurrences
        matches = list(re.finditer(r'\b' + escaped_text + r'\b', text))
        if not matches:
            # Try without word boundaries for partial matches
            matches = list(re.finditer(escaped_text, text))
            
        positions = [(match.start(), match.end()) for match in matches]
        
        if positions:
            span_positions[span_text] = positions
            
    return span_positions


def add_spans_to_annotations(annotations: List[Dict], raw_text: str) -> List[Dict]:
    """
    Add correct span positions to annotations based on the text content.
    Handles multiple occurrences of the same text and overlapping entities.
    
    Args:
        annotations: List of annotations without span positions
        raw_text: The full text to search in
    
    Returns:
        List[Dict]: Annotations with added span positions
    """
    # Group annotations by text
    text_to_annotations = {}
    for ann in annotations:
        text = ann.get('text', '')
        if text not in text_to_annotations:
            text_to_annotations[text] = []
        text_to_annotations[text].append(ann)
    
    # Find all occurrences of each unique text
    span_positions = find_spans_in_text(raw_text, list(text_to_annotations.keys()))
    
    # Create final annotations with positions
    final_annotations = []
    
    # Track positions we've already used to detect overlapping entities
    used_positions = set()
    
    for text, positions in span_positions.items():
        anns_for_text = text_to_annotations[text]
        
        # Check if we have multiple annotations for the same text (potentially different tags)
        if len(anns_for_text) > 1:
            # Check if we have enough positions for all annotations
            if len(positions) >= len(anns_for_text):
                # Assign each annotation to a unique position
                for i, ann in enumerate(anns_for_text):
                    start, stop = positions[i]
                    new_ann = ann.copy()
                    new_ann['start'] = start
                    new_ann['stop'] = stop
                    
                    # Track that we used this position
                    pos_key = (start, stop)
                    if pos_key in used_positions:
                        new_ann['overlapping_entity'] = True
                    else:
                        used_positions.add(pos_key)
                        
                    final_annotations.append(new_ann)
            else:
                # If we have more annotations than positions,
                # we need to recognize that the same entity can have multiple tags
                for i, position in enumerate(positions):
                    start, stop = position
                    pos_key = (start, stop)
                    
                    # Mark all annotations for this position as overlapping entities
                    for j, ann in enumerate(anns_for_text):
                        new_ann = ann.copy()
                        new_ann['start'] = start
                        new_ann['stop'] = stop
                        
                        if pos_key in used_positions or j > 0:
                            new_ann['overlapping_entity'] = True
                        else:
                            used_positions.add(pos_key)
                            
                        final_annotations.append(new_ann)
        else:
            # We only have one annotation for this text
            ann = anns_for_text[0]
            
            # If we have multiple positions for this text, use the first one
            if positions:
                start, stop = positions[0]
                new_ann = ann.copy()
                new_ann['start'] = start
                new_ann['stop'] = stop
                
                # Check for overlaps
                pos_key = (start, stop)
                if pos_key in used_positions:
                    new_ann['overlapping_entity'] = True
                else:
                    used_positions.add(pos_key)
                    
                final_annotations.append(new_ann)
    
    # Check for any annotations that didn't get positions
    for text, anns in text_to_annotations.items():
        if text not in span_positions:
            for ann in anns:
                # Create a copy with error flag
                new_ann = ann.copy()
                new_ann['start'] = -1
                new_ann['stop'] = -1
                new_ann['position_not_found'] = True
                final_annotations.append(new_ann)
    
    return final_annotations


def annotate_article(
    article_id: str, 
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    num_examples: int = 3,
    save_result: bool = True,
    use_extended_scheme: bool = True,
    scheme_name: Optional[str] = None
) -> List[Dict]:
    """
    Annotate a policy article using the LLM.
    
    Args:
        article_id: The ID of the article to annotate
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        num_examples: Number of few-shot examples to include
        save_result: Whether to save the results to a file
        use_extended_scheme: Whether to use the extended scheme with examples
        scheme_name: Optional name of a specific coding scheme file to use
    
    Returns:
        List[Dict]: The generated annotations
    """
    # Load necessary data
    raw_text = load_raw_text(article_id)
    
    # Load the appropriate coding scheme
    if use_extended_scheme and not scheme_name:
        scheme_name = "Coding_Scheme_Extended"
    coding_scheme = load_coding_scheme(scheme_name=scheme_name)
    
    # Load few-shot examples, excluding the target article
    few_shot_examples = load_few_shot_examples(
        num_examples=num_examples,
        layers=layers,
        tagsets=tagsets,
        exclude_article_ids=[article_id],
        use_extended_scheme=use_extended_scheme
    )
    
    # Prepare the prompt
    prompt = prepare_annotation_prompt(
        raw_text=raw_text,
        coding_scheme=coding_scheme,
        layers=layers,
        tagsets=tagsets,
        few_shot_examples=few_shot_examples,
        use_extended_scheme=use_extended_scheme
    )
    
    # Call the LLM API
    response = call_anthropic_api(prompt)
    
    # Parse the basic annotations (without positions)
    basic_annotations = parse_annotations_from_response(response)
    
    # Add span positions to annotations
    annotations = add_spans_to_annotations(basic_annotations, raw_text)
    
    # Log any annotations with position issues
    position_issues = [ann for ann in annotations if ann.get('position_not_found', False)]
    if position_issues:
        print(f"Warning: {len(position_issues)} annotations could not be positioned in the text.")
        for ann in position_issues:
            print(f"  - {ann.get('layer')}/{ann.get('feature')}/{ann.get('tag')}: '{ann.get('text')}'")
    
    # Save the results if requested
    if save_result:
        save_generated_annotations(article_id, annotations)
    
    return annotations 