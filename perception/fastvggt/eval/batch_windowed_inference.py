#!/usr/bin/env python3
"""
Batch processing script for windowed inference on multiple sequences.
Processes all sequences in a directory with different parameter combinations.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import glob
import time


def find_sequence_directories(base_path: Path) -> list:
    """
    Find all sequence directories containing images.
    
    Args:
        base_path: Base directory to search
        
    Returns:
        List of sequence directory paths
    """
    sequence_dirs = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            sequence_dirs.append(item)
    
    return sorted(sequence_dirs)


def run_single_sequence(data_path: Path, output_base: Path, 
                       window_size: int, overlap: int,
                       smoothing_factor: float,
                       alignment_method: str) -> dict:
    """
    Run windowed inference on a single sequence.
    
    Args:
        data_path: Path to sequence directory
        output_base: Base output directory
        window_size: Window size
        overlap: Overlap size
        smoothing_factor: Smoothing factor
        alignment_method: Alignment method
        
    Returns:
        Dictionary with processing results
    """
    sequence_name = data_path.name
    output_path = output_base / f"{sequence_name}_w{window_size}_o{overlap}_s{smoothing_factor:.1f}_{alignment_method}"
    
    cmd = [
        sys.executable, "windowed_inference.py",
        "--data_dir", str(data_path),
        "--out_dir", str(output_path),
        "--window_size", str(window_size),
        "--overlap", str(overlap),
        "--smoothing_factor", str(smoothing_factor),
        "--alignment_method", alignment_method
    ]
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if result.returncode == 0:
            # Try to load processing metadata
            metadata_path = output_path / "processing_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            return {
                'success': True,
                'sequence_name': sequence_name,
                'output_path': str(output_path),
                'processing_time': processing_time,
                'metadata': metadata,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            return {
                'success': False,
                'sequence_name': sequence_name,
                'processing_time': processing_time,
                'error': result.stderr,
                'stdout': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'sequence_name': sequence_name,
            'error': "Processing timeout (1 hour)",
            'processing_time': 3600
        }
    except Exception as e:
        return {
            'success': False,
            'sequence_name': sequence_name,
            'error': str(e),
            'processing_time': 0
        }


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple sequences with windowed inference"
    )
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Base path containing sequence directories"
    )
    
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Base output directory"
    )
    
    parser.add_argument(
        "--window_sizes",
        type=int,
        nargs='+',
        default=[10, 20],
        help="Window sizes to test (default: [10, 20])"
    )
    
    parser.add_argument(
        "--overlaps",
        type=int,
        nargs='+',
        default=[3, 5],
        help="Overlap sizes to test (default: [3, 5])"
    )
    
    parser.add_argument(
        "--smoothing_factors",
        type=float,
        nargs='+',
        default=[0.0, 0.1, 0.2],
        help="Smoothing factors to test (default: [0.0, 0.1, 0.2])"
    )
    
    parser.add_argument(
        "--alignment_methods",
        type=str,
        nargs='+',
        choices=['icp', 'translation_only', 'rotation_only'],
        default=['icp', 'translation_only'],
        help="Alignment methods to test"
    )
    
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process (default: all)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action='store_true',
        help="Skip sequences that already have results"
    )
    
    args = parser.parse_args()
    
    # Find sequence directories
    print(f"🔍 Finding sequences in: {args.data_dir}")
    sequence_dirs = find_sequence_directories(args.data_dir)
    
    if not sequence_dirs:
        print(f"❌ No sequence directories found in {args.data_dir}")
        return
    
    if args.max_sequences:
        sequence_dirs = sequence_dirs[:args.max_sequences]


    print(f"📁 Found {len(sequence_dirs)} sequences:")
    for seq_dir in sequence_dirs:
        image_count = len(list(seq_dir.glob("*rgb.jpg")))
        print(f"  - {seq_dir.name}: {image_count} images")
    
    # Create output base directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    param_combinations = []
    for window_size in args.window_sizes:
        for overlap in args.overlaps:
            if overlap >= window_size:
                continue  # Skip invalid combinations
            for smoothing_factor in args.smoothing_factors:
                for alignment_method in args.alignment_methods:
                    param_combinations.append({
                        'window_size': window_size,
                        'overlap': overlap,
                        'smoothing_factor': smoothing_factor,
                        'alignment_method': alignment_method
                    })
    
    print(f"⚙️  Testing {len(param_combinations)} parameter combinations:")
    for i, params in enumerate(param_combinations):
        print(f"  {i+1}. w{params['window_size']}_o{params['overlap']}_s{params['smoothing_factor']:.1f}_{params['alignment_method']}")
    
    total_jobs = len(sequence_dirs) * len(param_combinations)
    print(f"🚀 Total jobs: {total_jobs}")
    
    # Process all combinations
    results = []
    job_count = 0
    
    for seq_dir in tqdm(sequence_dirs, desc="Sequences"):
        for params in param_combinations:
            job_count += 1
            
            # Check if already processed
            if args.skip_existing:
                expected_output = args.out_dir / f"{seq_dir.name}_w{params['window_size']}_o{params['overlap']}_s{params['smoothing_factor']:.1f}_{params['alignment_method']}"
                if expected_output.exists() and (expected_output / "processing_metadata.json").exists():
                    print(f"⏭️  Skipping {seq_dir.name} with params {params} (already exists)")
                    continue
            
            print(f"\n🔄 Processing {job_count}/{total_jobs}: {seq_dir.name}")
            print(f"   Parameters: {params}")
            
            result = run_single_sequence(
                seq_dir,
                args.out_dir,
                **params
            )
            
            results.append(result)
            
            if result['success']:
                print(f"✅ Success: {result['processing_time']:.1f}s")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    # Save batch results
    results_path = args.out_dir / "batch_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {results_path}")
    
    # Print best results for each sequence
    if successful > 0:
        print(f"\n🏆 Best results per sequence:")
        sequence_bests = {}
        
        for result in results:
            if result['success'] and 'metadata' in result:
                seq_name = result['sequence_name']
                metadata = result['metadata']
                
                if seq_name not in sequence_bests:
                    sequence_bests[seq_name] = result
                else:
                    # Compare based on trajectory statistics (lower std is better)
                    current_std = sum(metadata.get('processing_stats', {}).get('after', {}).get('position_std', [0, 0, 0]))
                    best_std = sum(sequence_bests[seq_name]['metadata'].get('processing_stats', {}).get('after', {}).get('position_std', [0, 0, 0]))
                    
                    if current_std < best_std:
                        sequence_bests[seq_name] = result
        
        for seq_name, best_result in sequence_bests.items():
            params = best_result['metadata']
            print(f"  {seq_name}: {params.get('window_size')}w_{params.get('overlap')}o_{params.get('smoothing_factor'):.1f}s_{params.get('alignment_method')}")


if __name__ == "__main__":
    main()
