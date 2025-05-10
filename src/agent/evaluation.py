"""
Evaluation module for comparing LLM-generated annotations with human-curated ones.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from collections import defaultdict, Counter

# Import utility functions
from src.agent.utils import (
    get_project_root, 
    load_curated_annotations,
    filter_annotations,
    load_coder_annotations
)


def remove_metadata_fields(annotations: List[Dict]) -> List[Dict]:
    """
    Remove metadata fields from annotations.
    
    Args:
        annotations: List of annotation dictionaries
    
    Returns:
        List[Dict]: Cleaned annotations
    """
    cleaned = []
    for ann in annotations:
        ann_copy = ann.copy()
        # Remove standard metadata fields
        if 'span_id' in ann_copy:
            del ann_copy['span_id']
        if 'tokens' in ann_copy:
            del ann_copy['tokens']
        
        # Remove our custom position flags
        if 'position_not_found' in ann_copy:
            del ann_copy['position_not_found']
        if 'duplicate_position' in ann_copy:
            del ann_copy['duplicate_position']
        if 'overlapping_entity' in ann_copy:
            del ann_copy['overlapping_entity']
            
        cleaned.append(ann_copy)
    return cleaned


def filter_valid_annotations(annotations: List[Dict]) -> List[Dict]:
    """
    Filter out annotations with invalid positions.
    
    Args:
        annotations: List of annotation dictionaries
    
    Returns:
        List[Dict]: List of annotations with valid positions
    """
    return [ann for ann in annotations if not ann.get('position_not_found', False)]


def calculate_span_overlap(span1: Dict, span2: Dict, tolerance: int = 0) -> float:
    """
    Calculate the overlap between two spans, with optional tolerance for boundaries.
    
    Args:
        span1: First span dictionary with start and stop
        span2: Second span dictionary with start and stop
        tolerance: Number of characters to allow for boundary tolerance (default: 0)
    
    Returns:
        float: Overlap coefficient [0-1]
    """
    # Get the start and stop positions with tolerance
    start1 = max(0, span1['start'] - tolerance)
    stop1 = span1['stop'] + tolerance
    start2 = max(0, span2['start'] - tolerance)
    stop2 = span2['stop'] + tolerance
    
    # Calculate intersection
    intersection = max(0, min(stop1, stop2) - max(start1, start2))
    
    # Calculate span lengths
    span1_length = span1['stop'] - span1['start']  # Original lengths without tolerance
    span2_length = span2['stop'] - span2['start']
    
    # Use the smaller span as the denominator (overlap coefficient)
    denominator = min(span1_length, span2_length)
    
    # Handle edge case of zero-length spans
    if denominator == 0:
        return 0.0
    
    return intersection / denominator


def find_matching_spans(generated: List[Dict], curated: List[Dict], 
                       threshold: float = 0.5, tolerance: int = 0) -> List[Tuple[Dict, Dict, float]]:
    """
    Find matching spans between generated and curated annotations.
    Takes into account overlapping entities and multiple tags for the same span.
    
    Args:
        generated: List of generated annotation dictionaries
        curated: List of curated annotation dictionaries
        threshold: Minimum overlap required to consider spans matching
        tolerance: Number of characters to allow for boundary tolerance
    
    Returns:
        List[Tuple[Dict, Dict, float]]: List of (generated, curated, overlap) tuples
    """
    matches = []
    
    for gen_ann in generated:
        for cur_ann in curated:
            overlap = calculate_span_overlap(gen_ann, cur_ann, tolerance=tolerance)
            if overlap >= threshold:
                matches.append((gen_ann, cur_ann, overlap))
    
    # Sort by overlap in descending order
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # For overlapping entities, we need a more sophisticated matching approach
    # that allows multiple matches for the same span, as long as the tags differ
    
    # First, group annotations by position
    gen_by_pos = defaultdict(list)
    cur_by_pos = defaultdict(list)
    
    for gen_ann in generated:
        key = (gen_ann['start'], gen_ann['stop'])
        gen_by_pos[key].append(gen_ann)
    
    for cur_ann in curated:
        key = (cur_ann['start'], cur_ann['stop'])
        cur_by_pos[key].append(cur_ann)
    
    # Now match based on positions and tags
    final_matches = []
    matched_gen_id = set()
    matched_cur_id = set()
    
    for (gen_ann, cur_ann, overlap) in matches:
        gen_id = id(gen_ann)
        cur_id = id(cur_ann)
        
        # Check if these positions already have a match with the same tag
        gen_pos = (gen_ann['start'], gen_ann['stop'])
        cur_pos = (cur_ann['start'], cur_ann['stop'])
        
        # If the tags match, prefer to match them
        if (gen_ann['tag'] == cur_ann['tag'] and 
            gen_id not in matched_gen_id and 
            cur_id not in matched_cur_id):
            final_matches.append((gen_ann, cur_ann, overlap))
            matched_gen_id.add(gen_id)
            matched_cur_id.add(cur_id)
        
        # If the positions match and we have overlapping entities, match them
        elif (gen_ann.get('overlapping_entity', False) or 
              cur_ann.get('overlapping_entity', False) or
              (gen_pos == cur_pos) and 
              gen_id not in matched_gen_id and 
              cur_id not in matched_cur_id):
            final_matches.append((gen_ann, cur_ann, overlap))
            matched_gen_id.add(gen_id)
            matched_cur_id.add(cur_id)
        
        # Otherwise, use the standard non-overlap approach
        elif gen_id not in matched_gen_id and cur_id not in matched_cur_id:
            final_matches.append((gen_ann, cur_ann, overlap))
            matched_gen_id.add(gen_id)
            matched_cur_id.add(cur_id)
    
    return final_matches


def calculate_precision_recall_f1(matches: List[Tuple[Dict, Dict, float]], 
                                generated: List[Dict], 
                                curated: List[Dict]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for span matching.
    
    Args:
        matches: List of (generated, curated, overlap) tuples
        generated: List of all generated annotation dictionaries
        curated: List of all curated annotation dictionaries
    
    Returns:
        Dict[str, float]: Dictionary with precision, recall, and F1 scores
    """
    if not curated:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': len(generated),
            'false_negatives': 0
        }
    
    if not generated:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(curated)
        }
    
    true_positives = len(matches)
    false_positives = len(generated) - true_positives
    false_negatives = len(curated) - true_positives
    
    precision = true_positives / len(generated) if generated else 0.0
    recall = true_positives / len(curated) if curated else 0.0
    
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def tag_agreement(matches: List[Tuple[Dict, Dict, float]]) -> Dict[str, Any]:
    """
    Calculate tag agreement metrics for matched spans.
    
    Args:
        matches: List of (generated, curated, overlap) tuples
    
    Returns:
        Dict[str, Any]: Dictionary with tag agreement metrics
    """
    if not matches:
        return {
            'layer_accuracy': 0.0,
            'feature_accuracy': 0.0,
            'tag_accuracy': 0.0,
            'full_match_accuracy': 0.0,
            'count': 0
        }
    
    layer_correct = 0
    feature_correct = 0
    tag_correct = 0
    full_match = 0
    
    for gen_ann, cur_ann, _ in matches:
        if gen_ann['layer'] == cur_ann['layer']:
            layer_correct += 1
        
        if gen_ann['feature'] == cur_ann['feature']:
            feature_correct += 1
        
        if gen_ann['tag'] == cur_ann['tag']:
            tag_correct += 1
        
        if (gen_ann['layer'] == cur_ann['layer'] and 
            gen_ann['feature'] == cur_ann['feature'] and 
            gen_ann['tag'] == cur_ann['tag']):
            full_match += 1
    
    return {
        'layer_accuracy': layer_correct / len(matches),
        'feature_accuracy': feature_correct / len(matches),
        'tag_accuracy': tag_correct / len(matches),
        'full_match_accuracy': full_match / len(matches),
        'count': len(matches)
    }


def create_confusion_matrix(matches: List[Tuple[Dict, Dict, float]]) -> Dict[str, Any]:
    """
    Create confusion matrices for layer, feature, and tag assignments.
    
    Args:
        matches: List of (generated, curated, overlap) tuples
    
    Returns:
        Dict[str, Any]: Dictionary with confusion matrices
    """
    # Initialize counters
    layer_confusion = defaultdict(Counter)
    feature_confusion = defaultdict(Counter)
    tag_confusion = defaultdict(Counter)
    
    for gen_ann, cur_ann, _ in matches:
        # Layer confusion
        layer_confusion[cur_ann['layer']][gen_ann['layer']] += 1
        
        # Feature confusion (only if layer matches)
        if gen_ann['layer'] == cur_ann['layer']:
            feature_confusion[cur_ann['feature']][gen_ann['feature']] += 1
        
        # Tag confusion (only if layer and feature match)
        if gen_ann['layer'] == cur_ann['layer'] and gen_ann['feature'] == cur_ann['feature']:
            tag_confusion[cur_ann['tag']][gen_ann['tag']] += 1
    
    return {
        'layer_confusion': {k: dict(v) for k, v in layer_confusion.items()},
        'feature_confusion': {k: dict(v) for k, v in feature_confusion.items()},
        'tag_confusion': {k: dict(v) for k, v in tag_confusion.items()}
    }


def analyze_overlapping_entities(
    generated_annotations: List[Dict], 
    curated_annotations: List[Dict]
) -> Dict[str, Any]:
    """
    Analyze how well the system identified overlapping entities (entities with multiple tags).
    
    Args:
        generated_annotations: List of generated annotation dictionaries
        curated_annotations: List of curated annotation dictionaries
    
    Returns:
        Dict[str, Any]: Overlapping entity analysis metrics
    """
    # Group curated annotations by position
    curated_by_pos = defaultdict(list)
    for ann in curated_annotations:
        pos_key = (ann.get('start', -1), ann.get('stop', -1))
        if pos_key[0] >= 0 and pos_key[1] >= 0:  # Valid positions only
            curated_by_pos[pos_key].append(ann)
    
    # Group generated annotations by position
    generated_by_pos = defaultdict(list)
    for ann in generated_annotations:
        if ann.get('position_not_found', False):
            continue  # Skip invalid positions
        pos_key = (ann.get('start', -1), ann.get('stop', -1))
        if pos_key[0] >= 0 and pos_key[1] >= 0:  # Valid positions only
            generated_by_pos[pos_key].append(ann)
    
    # Identify overlapping positions in curated annotations
    curated_overlapping = {pos: anns for pos, anns in curated_by_pos.items() if len(anns) > 1}
    
    # Check each overlapping position in curated annotations
    overlapping_matches = 0
    total_overlapping_tags = sum(len(anns) for anns in curated_overlapping.values())
    
    for pos, curated_anns in curated_overlapping.items():
        # Get tags for this position in curated annotations
        curated_tags = {(ann.get('layer', ''), ann.get('feature', ''), ann.get('tag', '')) 
                       for ann in curated_anns}
        
        # Check if we have matching generated annotations
        generated_anns = generated_by_pos.get(pos, [])
        if not generated_anns:
            # Check for nearby positions if exact match not found
            for gen_pos, gen_anns in generated_by_pos.items():
                gen_start, gen_stop = gen_pos
                cur_start, cur_stop = pos
                
                # Check for significant overlap
                overlap_len = min(gen_stop, cur_stop) - max(gen_start, cur_start)
                min_len = min(gen_stop - gen_start, cur_stop - cur_start)
                
                if min_len > 0 and overlap_len / min_len >= 0.5:
                    generated_anns = gen_anns
                    break
        
        # Get tags for this position in generated annotations
        generated_tags = {(ann.get('layer', ''), ann.get('feature', ''), ann.get('tag', '')) 
                         for ann in generated_anns}
        
        # Count matching tags
        for tag in curated_tags:
            if tag in generated_tags:
                overlapping_matches += 1
    
    # Calculate overlapping entity accuracy
    overlapping_accuracy = overlapping_matches / total_overlapping_tags if total_overlapping_tags > 0 else 0.0
    
    return {
        'curated_overlapping_positions': len(curated_overlapping),
        'total_overlapping_tags': total_overlapping_tags,
        'matched_overlapping_tags': overlapping_matches,
        'overlapping_accuracy': overlapping_accuracy
    }


def evaluate_annotations(
    generated_annotations: List[Dict],
    curated_annotations: List[Dict],
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    overlap_threshold: float = 0.5,
    tolerance: int = 0
) -> Dict[str, Any]:
    """
    Evaluate generated annotations against curated ones.
    
    Args:
        generated_annotations: List of generated annotation dictionaries
        curated_annotations: List of curated annotation dictionaries
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        overlap_threshold: Minimum overlap required to consider spans matching
        tolerance: Number of characters to allow for boundary tolerance
    
    Returns:
        Dict[str, Any]: Evaluation results
    """
    # Check for invalid positions
    invalid_positions = [ann for ann in generated_annotations if ann.get('position_not_found', False)]
    duplicate_positions = [ann for ann in generated_annotations if ann.get('duplicate_position', False)]
    overlapping_entities = [ann for ann in generated_annotations if ann.get('overlapping_entity', False)]
    
    # Filter out annotations with invalid positions
    valid_generated = filter_valid_annotations(generated_annotations)
    
    # Record the counts before cleaning
    total_generated = len(generated_annotations)
    valid_count = len(valid_generated)
    invalid_count = len(invalid_positions)
    duplicate_count = len(duplicate_positions)
    overlapping_count = len(overlapping_entities)
    
    # Clean annotations
    generated = remove_metadata_fields(valid_generated)
    curated = remove_metadata_fields(curated_annotations)
    
    # Filter by layers/tagsets if specified
    if layers or tagsets:
        generated = filter_annotations(generated, layers, tagsets)
        curated = filter_annotations(curated, layers, tagsets)
    
    # Find matching spans
    matches = find_matching_spans(generated, curated, overlap_threshold, tolerance=tolerance)
    
    # Calculate metrics
    span_metrics = calculate_precision_recall_f1(matches, generated, curated)
    tag_metrics = tag_agreement(matches)
    confusion = create_confusion_matrix(matches)
    
    # Look at overlapping entity accuracy
    overlapping_metrics = analyze_overlapping_entities(
        generated_annotations,
        curated_annotations
    )
    
    # Count extra annotations (false positives)
    extra_annotations_count = span_metrics['false_positives']
    
    # Add position quality metrics
    position_metrics = {
        'total_annotations': total_generated,
        'valid_positions': valid_count,
        'invalid_positions': invalid_count,
        'duplicate_positions': duplicate_count,
        'overlapping_entities': overlapping_count,
        'position_accuracy': valid_count / total_generated if total_generated > 0 else 1.0,
        'extra_annotations': extra_annotations_count
    }
    
    # Combine all metrics
    results = {
        'span_identification': span_metrics,
        'tag_assignment': tag_metrics,
        'position_quality': position_metrics,
        'overlapping_entities': overlapping_metrics,
        'confusion_matrices': confusion,
        'summary': {
            'span_f1': span_metrics['f1'],
            'full_tag_accuracy': tag_metrics['full_match_accuracy'],
            'position_accuracy': position_metrics['position_accuracy'],
            'overlapping_accuracy': overlapping_metrics['overlapping_accuracy'],
            'extra_annotations': extra_annotations_count,
            'combined_score': (span_metrics['f1'] + tag_metrics['full_match_accuracy']) / 2 if matches else 0.0
        }
    }
    
    return results


def compare_coder_to_curated(
    article_id: str,
    coder_id: str,
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    tolerance: int = 0
) -> Dict[str, Any]:
    """
    Compare individual coder annotations against curated annotations.
    
    Args:
        article_id: ID of the article to evaluate
        coder_id: Coder ID to evaluate (e.g., "A", "B")
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        tolerance: Number of characters to allow for boundary tolerance
    
    Returns:
        Dict[str, Any]: Evaluation results
    """
    root_dir = get_project_root()
    
    # Load curated annotations
    curated = load_curated_annotations(article_id)
    
    # Load coder annotations
    coder_annotations = load_coder_annotations(article_id, coder_id)
    
    # Run evaluation
    eval_results = evaluate_annotations(
        generated_annotations=coder_annotations,  # Treat coder annotations as "generated"
        curated_annotations=curated,              # Compare against curated
        layers=layers,
        tagsets=tagsets,
        tolerance=tolerance
    )
    
    # Add metadata
    eval_results['metadata'] = {
        'article_id': article_id,
        'coder_id': coder_id,
        'layers_filter': layers,
        'tagsets_filter': tagsets,
        'tolerance': tolerance,
        'curated_count': len(curated),
        'coder_count': len(coder_annotations),
        'filtered_curated_count': len(filter_annotations(curated, layers, tagsets)) if layers or tagsets else len(curated),
        'filtered_coder_count': len(filter_annotations(coder_annotations, layers, tagsets)) if layers or tagsets else len(coder_annotations)
    }
    
    # Add absolute metrics
    filtered_curated = filter_annotations(curated, layers, tagsets) if layers or tagsets else curated
    filtered_coder = filter_annotations(coder_annotations, layers, tagsets) if layers or tagsets else coder_annotations
    
    # Count correct annotations (true positives)
    correct_annotations = eval_results['span_identification']['true_positives']
    extra_annotations = eval_results['span_identification']['false_positives']
    missed_annotations = eval_results['span_identification']['false_negatives']
    
    eval_results['absolute_metrics'] = {
        'total_curated': len(filtered_curated),
        'total_coder': len(filtered_coder),
        'correct_annotations': correct_annotations,
        'extra_annotations': extra_annotations,
        'missed_annotations': missed_annotations
    }
    
    return eval_results


def get_absolute_metrics(results: Dict[str, Any]) -> Dict[str, int]:
    """
    Extract absolute metrics from evaluation results.
    
    Args:
        results: Evaluation results from evaluate_article or compare_coder_to_curated
    
    Returns:
        Dict[str, int]: Dictionary with absolute count metrics
    """
    if 'absolute_metrics' in results:
        # Already has absolute metrics
        return results['absolute_metrics']
    
    # For regular evaluation results
    correct_annotations = results['span_identification']['true_positives']
    extra_annotations = results['span_identification']['false_positives']
    missed_annotations = results['span_identification']['false_negatives']
    
    # Get total counts from metadata or calculate them
    if 'metadata' in results:
        if 'filtered_curated_count' in results['metadata']:
            total_curated = results['metadata']['filtered_curated_count']
        else:
            total_curated = correct_annotations + missed_annotations
            
        if 'filtered_generated_count' in results['metadata']:
            total_generated = results['metadata']['filtered_generated_count']
        else:
            total_generated = correct_annotations + extra_annotations
    else:
        total_curated = correct_annotations + missed_annotations
        total_generated = correct_annotations + extra_annotations
    
    return {
        'total_curated': total_curated,
        'total_generated': total_generated,
        'correct_annotations': correct_annotations,
        'extra_annotations': extra_annotations,
        'missed_annotations': missed_annotations
    }


def evaluate_article(
    article_id: str,
    generated_path: Optional[str] = None,
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    save_results: bool = True,
    tolerance: int = 0,
    use_coder_annotations: bool = False,
    coder_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate LLM annotations for a specific article.
    
    Args:
        article_id: ID of the article to evaluate
        generated_path: Optional path to generated annotations (default: Generated_Annotations.json)
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        save_results: Whether to save evaluation results to a file
        tolerance: Number of characters to allow for boundary tolerance
        use_coder_annotations: Whether to evaluate against individual coder annotations 
                             instead of curated annotations
        coder_id: Optional coder ID to evaluate against (e.g., "A", "B")
                 If None and use_coder_annotations is True, evaluates against all coders
    
    Returns:
        Dict[str, Any]: Evaluation results
    """
    root_dir = get_project_root()
    article_dir = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id)
    
    # Load generated annotations
    if generated_path is None:
        generated_path = os.path.join(article_dir, 'Generated_Annotations.json')
    
    try:
        with open(generated_path, 'r') as f:
            generated = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Generated annotations file not found at {generated_path}")
    
    results = {}
    
    # Determine which annotations to evaluate against
    if use_coder_annotations:
        # Evaluate against individual coder annotations
        try:
            if coder_id is not None:
                # Evaluate against a specific coder
                coder_annotations = load_coder_annotations(article_id, coder_id)
                # Run evaluation
                eval_results = evaluate_annotations(
                    generated_annotations=generated,
                    curated_annotations=coder_annotations,
                    layers=layers,
                    tagsets=tagsets,
                    tolerance=tolerance
                )
                
                # Add metadata
                eval_results['metadata'] = {
                    'article_id': article_id,
                    'coder_id': coder_id,
                    'layers_filter': layers,
                    'tagsets_filter': tagsets,
                    'tolerance': tolerance,
                    'generated_count': len(generated),
                    'valid_generated_count': len(filter_valid_annotations(generated)),
                    'coder_count': len(coder_annotations),
                    'filtered_coder_count': len(filter_annotations(coder_annotations, layers, tagsets)) if layers or tagsets else len(coder_annotations),
                    'filtered_generated_count': len(filter_annotations(filter_valid_annotations(generated), layers, tagsets)) if layers or tagsets else len(filter_valid_annotations(generated))
                }
                
                results = eval_results
            else:
                # Evaluate against all coders and compile results
                all_coders = load_coder_annotations(article_id)
                all_results = {}
                
                for coder, annotations in all_coders.items():
                    # Run evaluation for this coder
                    coder_results = evaluate_annotations(
                        generated_annotations=generated,
                        curated_annotations=annotations,
                        layers=layers,
                        tagsets=tagsets,
                        tolerance=tolerance
                    )
                    
                    # Add metadata
                    coder_results['metadata'] = {
                        'article_id': article_id,
                        'coder_id': coder,
                        'layers_filter': layers,
                        'tagsets_filter': tagsets,
                        'tolerance': tolerance,
                        'generated_count': len(generated),
                        'valid_generated_count': len(filter_valid_annotations(generated)),
                        'coder_count': len(annotations),
                        'filtered_coder_count': len(filter_annotations(annotations, layers, tagsets)) if layers or tagsets else len(annotations),
                        'filtered_generated_count': len(filter_annotations(filter_valid_annotations(generated), layers, tagsets)) if layers or tagsets else len(filter_valid_annotations(generated))
                    }
                    
                    all_results[coder] = coder_results
                
                # Calculate average metrics across all coders
                avg_metrics = {
                    'span_identification': {
                        'precision': np.mean([r['span_identification']['precision'] for r in all_results.values()]),
                        'recall': np.mean([r['span_identification']['recall'] for r in all_results.values()]),
                        'f1': np.mean([r['span_identification']['f1'] for r in all_results.values()]),
                        'true_positives': sum([r['span_identification']['true_positives'] for r in all_results.values()]),
                        'false_positives': sum([r['span_identification']['false_positives'] for r in all_results.values()]),
                        'false_negatives': sum([r['span_identification']['false_negatives'] for r in all_results.values()])
                    },
                    'tag_assignment': {
                        'layer_accuracy': np.mean([r['tag_assignment']['layer_accuracy'] for r in all_results.values() if r['tag_assignment']['count'] > 0]),
                        'feature_accuracy': np.mean([r['tag_assignment']['feature_accuracy'] for r in all_results.values() if r['tag_assignment']['count'] > 0]),
                        'tag_accuracy': np.mean([r['tag_assignment']['tag_accuracy'] for r in all_results.values() if r['tag_assignment']['count'] > 0]),
                        'full_match_accuracy': np.mean([r['tag_assignment']['full_match_accuracy'] for r in all_results.values() if r['tag_assignment']['count'] > 0]),
                        'count': sum([r['tag_assignment']['count'] for r in all_results.values()])
                    },
                    'summary': {
                        'span_f1': np.mean([r['summary']['span_f1'] for r in all_results.values()]),
                        'full_tag_accuracy': np.mean([r['summary']['full_tag_accuracy'] for r in all_results.values() if r['tag_assignment']['count'] > 0]),
                        'extra_annotations': np.mean([r['summary']['extra_annotations'] for r in all_results.values()]),
                        'combined_score': np.mean([r['summary']['combined_score'] for r in all_results.values()])
                    },
                    'metadata': {
                        'article_id': article_id,
                        'coders_evaluated': list(all_results.keys()),
                        'layers_filter': layers,
                        'tagsets_filter': tagsets,
                        'tolerance': tolerance
                    }
                }
                
                results = {
                    'individual_coders': all_results,
                    'average_metrics': avg_metrics
                }
        except Exception as e:
            raise Exception(f"Error evaluating against coder annotations: {e}")
    else:
        # Evaluate against curated annotations (default behavior)
        curated = load_curated_annotations(article_id)
        
        # Run evaluation
        eval_results = evaluate_annotations(
            generated_annotations=generated,
            curated_annotations=curated,
            layers=layers,
            tagsets=tagsets,
            tolerance=tolerance
        )
        
        # Add metadata
        eval_results['metadata'] = {
            'article_id': article_id,
            'layers_filter': layers,
            'tagsets_filter': tagsets,
            'tolerance': tolerance,
            'curated_count': len(curated),
            'generated_count': len(generated),
            'valid_generated_count': len(filter_valid_annotations(generated)),
            'filtered_curated_count': len(filter_annotations(curated, layers, tagsets)) if layers or tagsets else len(curated),
            'filtered_generated_count': len(filter_annotations(filter_valid_annotations(generated), layers, tagsets)) if layers or tagsets else len(filter_valid_annotations(generated))
        }
        
        results = eval_results
    
    # After creating results, add absolute metrics
    if 'absolute_metrics' not in results:
        # For standard evaluations (not coder vs. curated)
        if not (isinstance(results, dict) and 'average_metrics' in results):
            # This is a single evaluation result
            correct_annotations = results['span_identification']['true_positives']
            extra_annotations = results['span_identification']['false_positives']
            missed_annotations = results['span_identification']['false_negatives']
            
            results['absolute_metrics'] = {
                'total_curated': results['metadata'].get('filtered_curated_count', correct_annotations + missed_annotations),
                'total_generated': results['metadata'].get('filtered_generated_count', correct_annotations + extra_annotations),
                'correct_annotations': correct_annotations,
                'extra_annotations': extra_annotations,
                'missed_annotations': missed_annotations
            }
    
    # Save results if requested
    if save_results:
        result_filename = 'Evaluation_Results'
        if use_coder_annotations:
            if coder_id:
                result_filename += f'_Coder_{coder_id}'
            else:
                result_filename += '_All_Coders'
        
        if tolerance > 0:
            result_filename += f'_Tolerance_{tolerance}'
            
        results_path = os.path.join(article_dir, f'{result_filename}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def batch_evaluate(
    article_ids: List[str],
    layers: Optional[List[str]] = None,
    tagsets: Optional[List[str]] = None,
    save_results: bool = True,
    tolerance: int = 0,
    use_coder_annotations: bool = False,
    coder_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Batch evaluate multiple articles.
    
    Args:
        article_ids: List of article IDs to evaluate
        layers: Optional list of layers to focus on
        tagsets: Optional list of tagsets to focus on
        save_results: Whether to save evaluation results
        tolerance: Number of characters to allow for boundary tolerance
        use_coder_annotations: Whether to evaluate against individual coder annotations
                             instead of curated annotations
        coder_id: Optional coder ID to evaluate against (e.g., "A", "B")
                 If None and use_coder_annotations is True, evaluates against all coders
    
    Returns:
        Dict[str, Any]: Overall evaluation results
    """
    all_results = []
    overall_metrics = {
        'span_identification': {
            'precision': [],
            'recall': [],
            'f1': [],
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0
        },
        'tag_assignment': {
            'layer_accuracy': [],
            'feature_accuracy': [],
            'tag_accuracy': [],
            'full_match_accuracy': [],
            'count': 0
        },
        'position_quality': {
            'total_annotations': 0,
            'valid_positions': 0,
            'invalid_positions': 0,
            'duplicate_positions': 0,
            'position_accuracy': [],
            'extra_annotations': 0
        },
        'summary': {
            'span_f1': [],
            'full_tag_accuracy': [],
            'position_accuracy': [],
            'extra_annotations': [],
            'combined_score': []
        }
    }
    
    for article_id in article_ids:
        try:
            result = evaluate_article(
                article_id=article_id,
                layers=layers,
                tagsets=tagsets,
                save_results=save_results,
                tolerance=tolerance,
                use_coder_annotations=use_coder_annotations,
                coder_id=coder_id
            )
            
            # If using all coders without a specific ID, the result structure is different
            if use_coder_annotations and coder_id is None:
                # Use the average metrics from all coders
                avg_metrics = result['average_metrics']
                all_results.append(avg_metrics)
                
                # Aggregate metrics from average results
                overall_metrics['span_identification']['precision'].append(avg_metrics['span_identification']['precision'])
                overall_metrics['span_identification']['recall'].append(avg_metrics['span_identification']['recall'])
                overall_metrics['span_identification']['f1'].append(avg_metrics['span_identification']['f1'])
                overall_metrics['span_identification']['true_positives'] += avg_metrics['span_identification']['true_positives']
                overall_metrics['span_identification']['false_positives'] += avg_metrics['span_identification']['false_positives']
                overall_metrics['span_identification']['false_negatives'] += avg_metrics['span_identification']['false_negatives']
                
                if avg_metrics['tag_assignment']['count'] > 0:
                    overall_metrics['tag_assignment']['layer_accuracy'].append(avg_metrics['tag_assignment']['layer_accuracy'])
                    overall_metrics['tag_assignment']['feature_accuracy'].append(avg_metrics['tag_assignment']['feature_accuracy'])
                    overall_metrics['tag_assignment']['tag_accuracy'].append(avg_metrics['tag_assignment']['tag_accuracy'])
                    overall_metrics['tag_assignment']['full_match_accuracy'].append(avg_metrics['tag_assignment']['full_match_accuracy'])
                    overall_metrics['tag_assignment']['count'] += avg_metrics['tag_assignment']['count']
                
                overall_metrics['summary']['span_f1'].append(avg_metrics['summary']['span_f1'])
                overall_metrics['summary']['full_tag_accuracy'].append(avg_metrics['summary']['full_tag_accuracy'])
                overall_metrics['summary']['extra_annotations'].append(avg_metrics['summary']['extra_annotations'])
                overall_metrics['summary']['combined_score'].append(avg_metrics['summary']['combined_score'])
            else:
                all_results.append(result)
                
                # Aggregate metrics
                overall_metrics['span_identification']['precision'].append(result['span_identification']['precision'])
                overall_metrics['span_identification']['recall'].append(result['span_identification']['recall'])
                overall_metrics['span_identification']['f1'].append(result['span_identification']['f1'])
                overall_metrics['span_identification']['true_positives'] += result['span_identification']['true_positives']
                overall_metrics['span_identification']['false_positives'] += result['span_identification']['false_positives']
                overall_metrics['span_identification']['false_negatives'] += result['span_identification']['false_negatives']
                
                if result['tag_assignment']['count'] > 0:
                    overall_metrics['tag_assignment']['layer_accuracy'].append(result['tag_assignment']['layer_accuracy'])
                    overall_metrics['tag_assignment']['feature_accuracy'].append(result['tag_assignment']['feature_accuracy'])
                    overall_metrics['tag_assignment']['tag_accuracy'].append(result['tag_assignment']['tag_accuracy'])
                    overall_metrics['tag_assignment']['full_match_accuracy'].append(result['tag_assignment']['full_match_accuracy'])
                    overall_metrics['tag_assignment']['count'] += result['tag_assignment']['count']
                
                # Track position quality metrics
                overall_metrics['position_quality']['total_annotations'] += result['position_quality']['total_annotations']
                overall_metrics['position_quality']['valid_positions'] += result['position_quality']['valid_positions']
                overall_metrics['position_quality']['invalid_positions'] += result['position_quality']['invalid_positions']
                overall_metrics['position_quality']['duplicate_positions'] += result['position_quality']['duplicate_positions']
                overall_metrics['position_quality']['position_accuracy'].append(result['position_quality']['position_accuracy'])
                overall_metrics['position_quality']['extra_annotations'] += result['position_quality']['extra_annotations']
                
                overall_metrics['summary']['span_f1'].append(result['summary']['span_f1'])
                overall_metrics['summary']['full_tag_accuracy'].append(result['summary']['full_tag_accuracy'])
                overall_metrics['summary']['position_accuracy'].append(result['summary']['position_accuracy'])
                overall_metrics['summary']['extra_annotations'].append(result['summary']['extra_annotations'])
                overall_metrics['summary']['combined_score'].append(result['summary']['combined_score'])
            
        except Exception as e:
            print(f"Error evaluating article {article_id}: {e}")
    
    # Calculate average metrics
    batch_results = {
        'span_identification': {
            'precision': np.mean(overall_metrics['span_identification']['precision']) if overall_metrics['span_identification']['precision'] else 0.0,
            'recall': np.mean(overall_metrics['span_identification']['recall']) if overall_metrics['span_identification']['recall'] else 0.0,
            'f1': np.mean(overall_metrics['span_identification']['f1']) if overall_metrics['span_identification']['f1'] else 0.0,
            'true_positives': overall_metrics['span_identification']['true_positives'],
            'false_positives': overall_metrics['span_identification']['false_positives'],
            'false_negatives': overall_metrics['span_identification']['false_negatives']
        },
        'tag_assignment': {
            'layer_accuracy': np.mean(overall_metrics['tag_assignment']['layer_accuracy']) if overall_metrics['tag_assignment']['layer_accuracy'] else 0.0,
            'feature_accuracy': np.mean(overall_metrics['tag_assignment']['feature_accuracy']) if overall_metrics['tag_assignment']['feature_accuracy'] else 0.0,
            'tag_accuracy': np.mean(overall_metrics['tag_assignment']['tag_accuracy']) if overall_metrics['tag_assignment']['tag_accuracy'] else 0.0,
            'full_match_accuracy': np.mean(overall_metrics['tag_assignment']['full_match_accuracy']) if overall_metrics['tag_assignment']['full_match_accuracy'] else 0.0,
            'count': overall_metrics['tag_assignment']['count']
        },
        'position_quality': {
            'total_annotations': overall_metrics['position_quality']['total_annotations'],
            'valid_positions': overall_metrics['position_quality']['valid_positions'],
            'invalid_positions': overall_metrics['position_quality']['invalid_positions'],
            'duplicate_positions': overall_metrics['position_quality']['duplicate_positions'],
            'position_accuracy': np.mean(overall_metrics['position_quality']['position_accuracy']) if overall_metrics['position_quality']['position_accuracy'] else 1.0,
            'extra_annotations': overall_metrics['position_quality']['extra_annotations']
        },
        'summary': {
            'span_f1': np.mean(overall_metrics['summary']['span_f1']) if overall_metrics['summary']['span_f1'] else 0.0,
            'full_tag_accuracy': np.mean(overall_metrics['summary']['full_tag_accuracy']) if overall_metrics['summary']['full_tag_accuracy'] else 0.0,
            'position_accuracy': np.mean(overall_metrics['summary']['position_accuracy']) if overall_metrics['summary']['position_accuracy'] else 1.0,
            'extra_annotations': np.mean(overall_metrics['summary']['extra_annotations']) if overall_metrics['summary']['extra_annotations'] else 0.0,
            'combined_score': np.mean(overall_metrics['summary']['combined_score']) if overall_metrics['summary']['combined_score'] else 0.0
        },
        'metadata': {
            'article_count': len(all_results),
            'evaluated_articles': [result['metadata']['article_id'] for result in all_results if 'article_id' in result.get('metadata', {})],
            'layers_filter': layers,
            'tagsets_filter': tagsets,
            'tolerance': tolerance,
            'use_coder_annotations': use_coder_annotations,
            'coder_id': coder_id
        }
    }
    
    # Save overall results if requested
    if save_results and all_results:
        root_dir = get_project_root()
        
        filename_parts = []
        if layers:
            filename_parts.append(f"layers_{'_'.join(layers)}")
        if tagsets:
            filename_parts.append(f"tagsets_{'_'.join(tagsets)}")
        if tolerance > 0:
            filename_parts.append(f"tolerance_{tolerance}")
        if use_coder_annotations:
            if coder_id:
                filename_parts.append(f"coder_{coder_id}")
            else:
                filename_parts.append("all_coders")
        
        filter_str = "_".join(filename_parts)
        if filter_str:
            filter_str = "_" + filter_str
        
        results_path = os.path.join(root_dir, f'evaluation_batch{filter_str}.json')
        with open(results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
    
    return batch_results 