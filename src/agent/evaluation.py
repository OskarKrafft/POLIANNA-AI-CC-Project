"""
Evaluation module for policy text annotations.
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd

from src.agent.utils import get_project_root, load_coding_scheme


def is_text_match(curated_text: str, generated_text: str) -> bool:
    """
    Check if curated text is a substring of generated text.
    Both are normalized to lowercase for comparison.
    """
    return curated_text.lower() in generated_text.lower()


def evaluate_article(
    article_id: str,
    ignore_position: bool = True,
    generated_standard_path: Optional[str] = None,
    generated_extended_path: Optional[str] = None
) -> Dict:
    """
    Evaluate generated annotations against curated annotations.
    
    Args:
        article_id: The article ID to evaluate
        ignore_position: If True, ignore start/stop positions and use text matching
        generated_standard_path: Path to standard scheme annotations (optional)
        generated_extended_path: Path to extended scheme annotations (optional)
    
    Returns:
        Dictionary containing evaluation results
    """
    # Get paths
    root_dir = get_project_root()
    article_dir = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id)
    
    # Load curated annotations
    curated_path = os.path.join(article_dir, 'Curated_Annotations.json')
    with open(curated_path, 'r') as f:
        curated_annotations = json.load(f)
    
    # Load generated annotations
    results = {}
    
    # Standard scheme
    if generated_standard_path is None:
        generated_standard_path = os.path.join(article_dir, 'Generated_Annotations_Standard.json')
    
    if os.path.exists(generated_standard_path):
        with open(generated_standard_path, 'r') as f:
            standard_annotations = json.load(f)
        results['standard'] = evaluate_annotations(curated_annotations, standard_annotations, ignore_position)
    
    # Extended scheme  
    if generated_extended_path is None:
        generated_extended_path = os.path.join(article_dir, 'Generated_Annotations_Extended.json')
        
    if os.path.exists(generated_extended_path):
        with open(generated_extended_path, 'r') as f:
            extended_annotations = json.load(f)
        results['extended'] = evaluate_annotations(curated_annotations, extended_annotations, ignore_position)
    
    return results


def evaluate_annotations(
    curated_annotations: List[Dict],
    generated_annotations: List[Dict], 
    ignore_position: bool = True
) -> Dict:
    """
    Evaluate generated annotations against curated annotations.
    
    Args:
        curated_annotations: List of curated annotation dictionaries
        generated_annotations: List of generated annotation dictionaries
        ignore_position: If True, use text matching instead of position matching
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Normalize generated annotations (fix layer format issues)
    normalized_generated = []
    for ann in generated_annotations:
        norm_ann = ann.copy()
        
        # Fix layer format if necessary
        layer = ann.get('layer', '')
        if layer == 'Actor':
            norm_ann['layer'] = 'Policydesigncharacteristics'
            norm_ann['feature'] = 'Actor'
        elif layer == 'Reference':
            norm_ann['layer'] = 'Policydesigncharacteristics'
            norm_ann['feature'] = 'Reference'
        elif layer == 'InstrumentType':
            norm_ann['layer'] = 'Instrumenttypes'
            norm_ann['feature'] = 'InstrumentType'
        
        # Include all annotations when ignoring position
        if ignore_position:
            normalized_generated.append(norm_ann)
        else:
            # Only include if position exists
            if ann.get('start', -1) != -1:
                normalized_generated.append(norm_ann)
    
    # Find unique tag-text combinations that are successfully matched
    matched_gen_indices = set()  # Track which generated annotations were matched
    
    # Count True Positives and False Negatives
    tp_overall = 0
    fn_overall = 0
    tp_by_layer = defaultdict(int)
    fn_by_layer = defaultdict(int)
    tp_by_tagset = defaultdict(int)
    fn_by_tagset = defaultdict(int)
    total_by_layer = defaultdict(int)
    total_by_tagset = defaultdict(int)
    
    # First pass: Find matches between curated and generated
    for curated_ann in curated_annotations:
        layer = curated_ann.get('layer', '')
        feature = curated_ann.get('feature', '')
        tag = curated_ann.get('tag', '')
        curated_text = curated_ann.get('text', '').lower().strip()
        
        total_by_layer[layer] += 1
        total_by_tagset[f"{layer}/{feature}"] += 1
        
        # Find if this curated annotation has a match
        found_match = False
        
        if ignore_position:
            for i, gen_ann in enumerate(normalized_generated):
                if gen_ann.get('tag', '') == tag:
                    gen_text = gen_ann.get('text', '').lower().strip()
                    if curated_text in gen_text:
                        found_match = True
                        matched_gen_indices.add(i)
                        break
        else:
            # Use exact position matching
            curated_start = curated_ann.get('start')
            curated_stop = curated_ann.get('stop')
            for i, gen_ann in enumerate(normalized_generated):
                if (gen_ann.get('tag', '') == tag and
                    gen_ann.get('start') == curated_start and 
                    gen_ann.get('stop') == curated_stop):
                    found_match = True
                    matched_gen_indices.add(i)
                    break
        
        if found_match:
            tp_overall += 1
            tp_by_layer[layer] += 1
            tp_by_tagset[f"{layer}/{feature}"] += 1
        else:
            fn_overall += 1
            fn_by_layer[layer] += 1
            fn_by_tagset[f"{layer}/{feature}"] += 1
    
    # Count False Positives (generated annotations that didn't match any curated)
    fp_overall = len(normalized_generated) - len(matched_gen_indices)
    fp_by_layer = defaultdict(int)
    fp_by_tagset = defaultdict(int)
    
    for i, gen_ann in enumerate(normalized_generated):
        if i not in matched_gen_indices:
            layer = gen_ann.get('layer', '')
            feature = gen_ann.get('feature', '')
            fp_by_layer[layer] += 1
            fp_by_tagset[f"{layer}/{feature}"] += 1
    
    # Calculate overall metrics
    precision = tp_overall / (tp_overall + fp_overall) if (tp_overall + fp_overall) > 0 else 0
    recall = tp_overall / (tp_overall + fn_overall) if (tp_overall + fn_overall) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate metrics by layer
    layer_metrics = {}
    for layer in set(list(total_by_layer.keys()) + list(fp_by_layer.keys())):
        tp = tp_by_layer[layer]
        fp = fp_by_layer[layer]
        fn = fn_by_layer[layer]
        
        layer_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        layer_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        layer_f1 = 2 * layer_precision * layer_recall / (layer_precision + layer_recall) if (layer_precision + layer_recall) > 0 else 0
        
        layer_metrics[layer] = {
            'precision': layer_precision,
            'recall': layer_recall,
            'f1_score': layer_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Calculate metrics by tagset
    tagset_metrics = {}
    for tagset in set(list(total_by_tagset.keys()) + list(fp_by_tagset.keys())):
        tp = tp_by_tagset[tagset]
        fp = fp_by_tagset[tagset]
        fn = fn_by_tagset[tagset]
        
        tagset_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        tagset_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        tagset_f1 = 2 * tagset_precision * tagset_recall / (tagset_precision + tagset_recall) if (tagset_precision + tagset_recall) > 0 else 0
        
        tagset_metrics[tagset] = {
            'precision': tagset_precision,
            'recall': tagset_recall,
            'f1_score': tagset_f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tp': tp_overall,
            'fp': fp_overall,
            'fn': fn_overall
        },
        'by_layer': layer_metrics,
        'by_tagset': tagset_metrics
    }


def create_results_table(results: Dict, scheme_name: str) -> pd.DataFrame:
    """
    Create a results table similar to the provided example.
    
    Args:
        results: Results dictionary from evaluate_annotations
        scheme_name: Name of the scheme ('Standard' or 'Extended')
    
    Returns:
        Pandas DataFrame with results table
    """
    rows = []
    
    # Overall results
    overall = results['overall']
    rows.append({
        'Entity Type': 'All Entities',
        'Precision': f"{overall['precision']:.2f}",
        'Recall': f"{overall['recall']:.2f}",
        'F1 Score': f"{overall['f1_score']:.2f}", 
        'TP': overall['tp'],
        'FP': overall['fp'],
        'FN': overall['fn']
    })
    
    # Add separator row
    rows.append({
        'Entity Type': f'Results by Layer',
        'Precision': '',
        'Recall': '',
        'F1 Score': '',
        'TP': '',
        'FP': '',
        'FN': ''
    })
    
    # Layer results
    for layer, metrics in results['by_layer'].items():
        # Simplify layer name for display
        display_layer = layer.replace('Policydesigncharacteristics', 'Policy')
        display_layer = display_layer.replace('Technologyandapplicationspecificity', 'Technology')
        display_layer = display_layer.replace('Instrumenttypes', 'Instrument')
        
        rows.append({
            'Entity Type': display_layer,
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1 Score': f"{metrics['f1_score']:.2f}",
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        })
    
    # Add separator row
    rows.append({
        'Entity Type': f'Results by Tagset',
        'Precision': '',
        'Recall': '',
        'F1 Score': '',
        'TP': '',
        'FP': '',
        'FN': ''
    })
    
    # Tagset results
    for tagset, metrics in results['by_tagset'].items():
        # Clean up tagset name for display
        display_tagset = tagset.replace('Policydesigncharacteristics/', '')
        display_tagset = display_tagset.replace('Technologyandapplicationspecificity/', '')
        display_tagset = display_tagset.replace('Instrumenttypes/', '')
        
        rows.append({
            'Entity Type': display_tagset,
            'Precision': f"{metrics['precision']:.2f}",
            'Recall': f"{metrics['recall']:.2f}",
            'F1 Score': f"{metrics['f1_score']:.2f}",
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn']
        })
    
    df = pd.DataFrame(rows)
    return df


def evaluate_multiple_articles(
    article_ids: List[str],
    ignore_position: bool = True
) -> Dict:
    """
    Evaluate multiple articles and compute average performance.
    
    Args:
        article_ids: List of article IDs to evaluate
        ignore_position: If True, ignore start/stop positions and use text matching
        
    Returns:
        Dictionary containing averaged evaluation results
    """
    all_standard_results = []
    all_extended_results = []
    
    for article_id in article_ids:
        try:
            results = evaluate_article(article_id, ignore_position)
            if 'standard' in results:
                all_standard_results.append(results['standard'])
            if 'extended' in results:
                all_extended_results.append(results['extended'])
        except Exception as e:
            print(f"Error evaluating article {article_id}: {e}")
            continue
    
    # Average the results
    averaged_results = {}
    
    if all_standard_results:
        averaged_results['standard'] = average_results(all_standard_results)
    
    if all_extended_results:
        averaged_results['extended'] = average_results(all_extended_results)
    
    return averaged_results


def average_results(results_list: List[Dict]) -> Dict:
    """
    Average a list of evaluation results.
    
    Args:
        results_list: List of evaluation result dictionaries
        
    Returns:
        Dictionary containing averaged results
    """
    if not results_list:
        return {}
    
    # Initialize totals
    total_tp = 0
    total_fp = 0  
    total_fn = 0
    
    layer_totals = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    tagset_totals = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Sum up all results
    for results in results_list:
        overall = results['overall']
        total_tp += overall['tp']
        total_fp += overall['fp']
        total_fn += overall['fn']
        
        for layer, metrics in results['by_layer'].items():
            layer_totals[layer]['tp'] += metrics['tp']
            layer_totals[layer]['fp'] += metrics['fp']
            layer_totals[layer]['fn'] += metrics['fn']
            
        for tagset, metrics in results['by_tagset'].items():
            tagset_totals[tagset]['tp'] += metrics['tp']
            tagset_totals[tagset]['fp'] += metrics['fp']
            tagset_totals[tagset]['fn'] += metrics['fn']
    
    # Calculate averaged metrics
    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1_score': f1, 'tp': tp, 'fp': fp, 'fn': fn}
    
    overall_metrics = calc_metrics(total_tp, total_fp, total_fn)
    
    layer_metrics = {}
    for layer, totals in layer_totals.items():
        layer_metrics[layer] = calc_metrics(totals['tp'], totals['fp'], totals['fn'])
    
    tagset_metrics = {}
    for tagset, totals in tagset_totals.items():
        tagset_metrics[tagset] = calc_metrics(totals['tp'], totals['fp'], totals['fn'])
    
    return {
        'overall': overall_metrics,
        'by_layer': layer_metrics,
        'by_tagset': tagset_metrics
    }


def print_evaluation_results(results: Dict, article_id: str = None):
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Results dictionary from evaluate_article
        article_id: Optional article ID for the header
    """
    if article_id:
        print(f"\nEvaluation Results for Article: {article_id}")
        print("=" * 50)
    
    for scheme_name, scheme_results in results.items():
        print(f"\n{scheme_name.capitalize()} Scheme Results:")
        print("-" * 30)
        
        df = create_results_table(scheme_results, scheme_name.capitalize())
        print(df.to_string(index=False))


def debug_evaluation(article_id: str, ignore_position: bool = True):
    """
    Debug function to understand what's happening in the evaluation.
    
    Args:
        article_id: The article ID to debug
        ignore_position: If True, ignore start/stop positions and use text matching
    """
    # Get paths
    root_dir = get_project_root()
    article_dir = os.path.join(root_dir, 'data', '03b_processed_to_json', article_id)
    
    # Load annotations
    curated_path = os.path.join(article_dir, 'Curated_Annotations.json')
    standard_path = os.path.join(article_dir, 'Generated_Annotations_Standard.json')
    extended_path = os.path.join(article_dir, 'Generated_Annotations_Extended.json')
    
    with open(curated_path, 'r') as f:
        curated = json.load(f)
    with open(standard_path, 'r') as f:
        standard = json.load(f)
    with open(extended_path, 'r') as f:
        extended = json.load(f)
    
    print(f"Debug Analysis for Article: {article_id}")
    print("=" * 60)
    
    print("\nCurated Annotations:")
    for i, ann in enumerate(curated):
        print(f"{i+1}. {ann.get('tag')}: '{ann.get('text')}' ({ann.get('start')}:{ann.get('stop')})")
    
    print("\nStandard Annotations:")
    for i, ann in enumerate(standard):
        print(f"{i+1}. {ann.get('tag')}: '{ann.get('text')}' ({ann.get('start', 'N/A')}:{ann.get('stop', 'N/A')})")
        if ann.get('position_not_found'):
            print(f"   [Position not found]")
    
    print("\nExtended Annotations:")
    for i, ann in enumerate(extended):
        print(f"{i+1}. {ann.get('tag')}: '{ann.get('text')}' ({ann.get('start', 'N/A')}:{ann.get('stop', 'N/A')})")
        if ann.get('position_not_found'):
            print(f"   [Position not found]")
    
    # Analyze matches and false positives for Standard scheme
    print("\nStandard Scheme Analysis:")
    print("-" * 40)
    matched_standard = set()
    
    print("True Positives:")
    for curated_ann in curated:
        tag = curated_ann.get('tag')
        curated_text = curated_ann.get('text', '').lower().strip()
        
        for i, gen_ann in enumerate(standard):
            gen_tag = gen_ann.get('tag')
            gen_text = gen_ann.get('text', '').lower().strip()
            
            # Handle layer format differences
            if gen_ann.get('layer') == 'Actor':
                gen_tag = gen_ann.get('tag')
            elif gen_ann.get('layer') == 'Reference':
                gen_tag = gen_ann.get('tag')
            
            if gen_tag == tag and curated_text in gen_text:
                print(f"  ✓ {tag}: '{curated_text}' matches '{gen_text}'")
                matched_standard.add(i)
                break
    
    print("False Negatives:")
    for curated_ann in curated:
        tag = curated_ann.get('tag')
        curated_text = curated_ann.get('text', '').lower().strip()
        found = False
        
        for gen_ann in standard:
            gen_tag = gen_ann.get('tag')
            
            # Handle layer format differences
            if gen_ann.get('layer') == 'Actor':
                gen_tag = gen_ann.get('tag')
            elif gen_ann.get('layer') == 'Reference':
                gen_tag = gen_ann.get('tag')
            
            gen_text = gen_ann.get('text', '').lower().strip()
            if gen_tag == tag and curated_text in gen_text:
                found = True
                break
        
        if not found:
            print(f"  ✗ {tag}: '{curated_text}' - NO MATCH")
    
    print("False Positives:")
    for i, gen_ann in enumerate(standard):
        if i not in matched_standard:
            print(f"  + {gen_ann.get('tag')}: '{gen_ann.get('text')}' - EXTRA")
    
    # Analyze matches and false positives for Extended scheme
    print("\nExtended Scheme Analysis:")
    print("-" * 40)
    matched_extended = set()
    
    print("True Positives:")
    for curated_ann in curated:
        tag = curated_ann.get('tag')
        curated_text = curated_ann.get('text', '').lower().strip()
        
        for i, gen_ann in enumerate(extended):
            if gen_ann.get('tag') == tag:
                gen_text = gen_ann.get('text', '').lower().strip()
                if curated_text in gen_text:
                    print(f"  ✓ {tag}: '{curated_text}' matches '{gen_text}'")
                    matched_extended.add(i)
                    break
    
    print("False Negatives:")
    for curated_ann in curated:
        tag = curated_ann.get('tag')
        curated_text = curated_ann.get('text', '').lower().strip()
        found = False
        
        for gen_ann in extended:
            if gen_ann.get('tag') == tag:
                gen_text = gen_ann.get('text', '').lower().strip()
                if curated_text in gen_text:
                    found = True
                    break
        
        if not found:
            print(f"  ✗ {tag}: '{curated_text}' - NO MATCH")
    
    print("False Positives:")
    for i, gen_ann in enumerate(extended):
        if i not in matched_extended:
            print(f"  + {gen_ann.get('tag')}: '{gen_ann.get('text')}' - EXTRA")
