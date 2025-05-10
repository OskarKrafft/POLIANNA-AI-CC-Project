#!/usr/bin/env python3
"""
Main module providing a command-line interface for the policy text annotation system.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict

from src.utils import (
    get_project_root, 
    get_articles_paths, 
    create_extended_coding_scheme
)
from src.annotation import annotate_article
from src.evaluation import evaluate_article, batch_evaluate, filter_valid_annotations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Policy Text Annotation System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Annotate command
    annotate_parser = subparsers.add_parser("annotate", help="Annotate articles")
    annotate_parser.add_argument("--article-id", required=True, help="Article ID to annotate")
    annotate_parser.add_argument("--layers", nargs="+", help="Specific layers to annotate")
    annotate_parser.add_argument("--tagsets", nargs="+", help="Specific tagsets to annotate")
    annotate_parser.add_argument("--examples", type=int, default=3, help="Number of few-shot examples")
    annotate_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    annotate_parser.add_argument("--use-extended", action="store_true", help="Use extended coding scheme with examples")
    annotate_parser.add_argument("--scheme-name", help="Specific coding scheme file to use")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate article annotations")
    evaluate_parser.add_argument("--article-id", required=True, help="Article ID to evaluate")
    evaluate_parser.add_argument("--generated-path", help="Path to generated annotations")
    evaluate_parser.add_argument("--layers", nargs="+", help="Specific layers to evaluate")
    evaluate_parser.add_argument("--tagsets", nargs="+", help="Specific tagsets to evaluate")
    evaluate_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    # Batch evaluate command
    batch_parser = subparsers.add_parser("batch", help="Batch evaluate multiple articles")
    batch_parser.add_argument("--article-ids", nargs="+", help="Article IDs to evaluate")
    batch_parser.add_argument("--all", action="store_true", help="Evaluate all available articles")
    batch_parser.add_argument("--layers", nargs="+", help="Specific layers to evaluate")
    batch_parser.add_argument("--tagsets", nargs="+", help="Specific tagsets to evaluate")
    batch_parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    # Create extended coding scheme command
    extend_parser = subparsers.add_parser("extend-scheme", help="Create extended coding scheme with examples")
    extend_parser.add_argument("--output-name", default="Coding_Scheme_Extended", 
                             help="Output filename (without extension)")
    extend_parser.add_argument("--min-occurrences", type=int, default=2, 
                             help="Minimum occurrences for an example to be included")
    extend_parser.add_argument("--max-examples", type=int, default=5, 
                             help="Maximum number of examples per tag")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available articles")
    
    return parser.parse_args()


def list_articles():
    """List all available articles."""
    articles = get_articles_paths()
    
    print(f"Found {len(articles)} articles:")
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['id']}")


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.command == "annotate":
        print(f"Annotating article {args.article_id}...")
        if args.layers:
            print(f"Focusing on layers: {', '.join(args.layers)}")
        if args.tagsets:
            print(f"Focusing on tagsets: {', '.join(args.tagsets)}")
        if args.use_extended:
            print("Using extended coding scheme with examples")
        if args.scheme_name:
            print(f"Using coding scheme: {args.scheme_name}")
        
        try:
            annotations = annotate_article(
                article_id=args.article_id,
                layers=args.layers,
                tagsets=args.tagsets,
                num_examples=args.examples,
                save_result=not args.no_save,
                use_extended_scheme=args.use_extended,
                scheme_name=args.scheme_name
            )
            
            # Count annotations with valid positions
            valid_annotations = filter_valid_annotations(annotations)
            total_annotations = len(annotations)
            valid_count = len(valid_annotations)
            position_accuracy = valid_count / total_annotations if total_annotations > 0 else 1.0
            
            # Count position issues
            invalid_positions = [ann for ann in annotations if ann.get('position_not_found', False)]
            duplicate_positions = [ann for ann in annotations if ann.get('duplicate_position', False)]
            overlapping_entities = [ann for ann in annotations if ann.get('overlapping_entity', False)]
            invalid_count = len(invalid_positions)
            duplicate_count = len(duplicate_positions)
            overlapping_count = len(overlapping_entities)
            
            print(f"Generated {total_annotations} annotations")
            print(f"Position accuracy: {position_accuracy:.2%} ({valid_count} valid, {invalid_count} invalid, {duplicate_count} with duplicate positions)")
            
            if overlapping_count > 0:
                print(f"Detected {overlapping_count} overlapping entities (multiple tags for the same position)")
                
                # Group overlapping entities by position
                pos_to_anns = defaultdict(list)
                for ann in overlapping_entities:
                    pos_key = (ann.get('start', -1), ann.get('stop', -1))
                    if pos_key[0] >= 0:  # Only valid positions
                        pos_to_anns[pos_key].append(ann)
                
                # Display overlapping entities grouped by position
                print("\nOverlapping entity details:")
                for pos, anns in pos_to_anns.items():
                    start, stop = pos
                    text = anns[0].get('text', '???')
                    print(f"  - '{text}' ({start}:{stop}) has {len(anns) + 1} possible tags:")
                    
                    # Find the non-overlapping annotation for this position
                    for ann in annotations:
                        if (ann.get('start', -1) == start and 
                            ann.get('stop', -1) == stop and 
                            not ann.get('overlapping_entity', False)):
                            print(f"    * {ann.get('layer', '???')}/{ann.get('feature', '???')}/{ann.get('tag', '???')} [primary]")
                            break
                    
                    # List the overlapping annotations
                    for ann in anns:
                        print(f"    * {ann.get('layer', '???')}/{ann.get('feature', '???')}/{ann.get('tag', '???')}")
            
            # Display tag distribution
            tag_counts = {}
            for ann in annotations:
                tag = ann.get('tag', 'Unknown')
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
            
            print("\nTag distribution:")
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {tag}: {count}")
            
            if not args.no_save:
                root_dir = get_project_root()
                save_path = os.path.join(
                    root_dir, 'data', '03b_processed_to_json', 
                    args.article_id, 'Generated_Annotations.json'
                )
                print(f"\nResults saved to {save_path}")
            
        except Exception as e:
            print(f"Error annotating article: {e}")
            sys.exit(1)
    
    elif args.command == "evaluate":
        print(f"Evaluating article {args.article_id}...")
        if args.layers:
            print(f"Focusing on layers: {', '.join(args.layers)}")
        if args.tagsets:
            print(f"Focusing on tagsets: {', '.join(args.tagsets)}")
        
        try:
            results = evaluate_article(
                article_id=args.article_id,
                generated_path=args.generated_path,
                layers=args.layers,
                tagsets=args.tagsets,
                save_results=not args.no_save
            )
            
            # Print summary results
            print("\nEvaluation Results:")
            
            # Position quality metrics
            print(f"Position Quality:")
            print(f"  - Valid positions: {results['position_quality']['valid_positions']} / {results['position_quality']['total_annotations']} ({results['position_quality']['position_accuracy']:.2%})")
            if results['position_quality']['invalid_positions'] > 0:
                print(f"  - Invalid positions: {results['position_quality']['invalid_positions']}")
            if results['position_quality']['duplicate_positions'] > 0:
                print(f"  - Duplicate positions: {results['position_quality']['duplicate_positions']}")
            if results['position_quality'].get('overlapping_entities', 0) > 0:
                print(f"  - Overlapping entities: {results['position_quality']['overlapping_entities']}")
            
            # Overlapping entity metrics
            if 'overlapping_entities' in results:
                print(f"\nOverlapping Entity Detection:")
                print(f"  - Curated overlapping positions: {results['overlapping_entities']['curated_overlapping_positions']}")
                print(f"  - Total overlapping tags: {results['overlapping_entities']['total_overlapping_tags']}")
                print(f"  - Correctly matched: {results['overlapping_entities']['matched_overlapping_tags']}")
                print(f"  - Accuracy: {results['overlapping_entities']['overlapping_accuracy']:.2%}")
            
            # Span identification metrics
            print(f"\nSpan Identification - Precision: {results['span_identification']['precision']:.4f}, "
                  f"Recall: {results['span_identification']['recall']:.4f}, "
                  f"F1: {results['span_identification']['f1']:.4f}")
            
            # Tag assignment metrics
            print(f"Tag Assignment - Layer: {results['tag_assignment']['layer_accuracy']:.4f}, "
                  f"Feature: {results['tag_assignment']['feature_accuracy']:.4f}, "
                  f"Tag: {results['tag_assignment']['tag_accuracy']:.4f}, "
                  f"Full Match: {results['tag_assignment']['full_match_accuracy']:.4f}")
            
            # Combined score
            print(f"Combined Score: {results['summary']['combined_score']:.4f}")
            
            if not args.no_save:
                root_dir = get_project_root()
                save_path = os.path.join(
                    root_dir, 'data', '03b_processed_to_json', 
                    args.article_id, 'Evaluation_Results.json'
                )
                print(f"Results saved to {save_path}")
            
        except Exception as e:
            print(f"Error evaluating article: {e}")
            sys.exit(1)
    
    elif args.command == "batch":
        if args.all:
            # Get all article IDs
            articles = get_articles_paths()
            article_ids = [article['id'] for article in articles]
        elif args.article_ids:
            article_ids = args.article_ids
        else:
            print("Error: Either --article-ids or --all must be specified")
            sys.exit(1)
        
        print(f"Batch evaluating {len(article_ids)} articles...")
        if args.layers:
            print(f"Focusing on layers: {', '.join(args.layers)}")
        if args.tagsets:
            print(f"Focusing on tagsets: {', '.join(args.tagsets)}")
        
        try:
            results = batch_evaluate(
                article_ids=article_ids,
                layers=args.layers,
                tagsets=args.tagsets,
                save_results=not args.no_save
            )
            
            # Print summary results
            print("\nBatch Evaluation Results:")
            print(f"Articles evaluated: {results['metadata']['article_count']}")
            
            # Position quality metrics
            print(f"Position Quality:")
            print(f"  - Valid positions: {results['position_quality']['valid_positions']} / {results['position_quality']['total_annotations']} ({results['position_quality']['position_accuracy']:.2%})")
            if results['position_quality']['invalid_positions'] > 0:
                print(f"  - Invalid positions: {results['position_quality']['invalid_positions']}")
            if results['position_quality']['duplicate_positions'] > 0:
                print(f"  - Duplicate positions: {results['position_quality']['duplicate_positions']}")
            if results['position_quality'].get('overlapping_entities', 0) > 0:
                print(f"  - Overlapping entities: {results['position_quality']['overlapping_entities']}")
            
            # Overlapping entity metrics
            if 'overlapping_entities' in results:
                print(f"\nOverlapping Entity Detection:")
                print(f"  - Curated overlapping positions: {results['overlapping_entities']['curated_overlapping_positions']}")
                print(f"  - Total overlapping tags: {results['overlapping_entities']['total_overlapping_tags']}")
                print(f"  - Correctly matched: {results['overlapping_entities']['matched_overlapping_tags']}")
                print(f"  - Accuracy: {results['overlapping_entities']['overlapping_accuracy']:.2%}")
            
            # Span identification metrics
            print(f"\nSpan Identification - Precision: {results['span_identification']['precision']:.4f}, "
                  f"Recall: {results['span_identification']['recall']:.4f}, "
                  f"F1: {results['span_identification']['f1']:.4f}")
            
            # Tag assignment metrics
            print(f"Tag Assignment - Layer: {results['tag_assignment']['layer_accuracy']:.4f}, "
                  f"Feature: {results['tag_assignment']['feature_accuracy']:.4f}, "
                  f"Tag: {results['tag_assignment']['tag_accuracy']:.4f}, "
                  f"Full Match: {results['tag_assignment']['full_match_accuracy']:.4f}")
            
            # Combined score
            print(f"Combined Score: {results['summary']['combined_score']:.4f}")
            
            if not args.no_save:
                root_dir = get_project_root()
                filter_str = ""
                if args.layers:
                    filter_str += f"_layers_{'_'.join(args.layers)}"
                if args.tagsets:
                    filter_str += f"_tagsets_{'_'.join(args.tagsets)}"
                
                save_path = os.path.join(root_dir, f'evaluation_batch{filter_str}.json')
                print(f"Results saved to {save_path}")
            
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
            sys.exit(1)
    
    elif args.command == "extend-scheme":
        print(f"Creating extended coding scheme '{args.output_name}'...")
        print(f"Minimum occurrences: {args.min_occurrences}, Maximum examples per tag: {args.max_examples}")
        
        try:
            output_path = create_extended_coding_scheme(
                output_name=args.output_name,
                min_occurrences=args.min_occurrences,
                max_examples=args.max_examples
            )
            print(f"Successfully created extended coding scheme at {output_path}")
        except Exception as e:
            print(f"Error creating extended coding scheme: {e}")
            sys.exit(1)
    
    elif args.command == "list":
        list_articles()
    
    else:
        print("Please specify a command. Run with --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main() 