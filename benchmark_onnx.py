"""
Benchmark PyTorch vs ONNX Runtime performance (speed and CUDA memory)
"""

import time
import numpy as np
import argparse
import logging
import torch
import cv2
import os
import json
from datetime import datetime
from nets.aliked import ALIKED
from nets.aliked_onnx import ALIKED_ONNX


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0


def benchmark_model(model, img_rgb, num_runs=10, warmup=3, track_memory=False):
    """
    Benchmark model inference time and memory usage
    
    Args:
        model: Model instance (PyTorch or ONNX)
        img_rgb: RGB image
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        track_memory: Whether to track GPU memory usage
    
    Returns:
        Dictionary with timing and memory statistics
    """
    # Reset CUDA memory stats if tracking
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Warmup
    for _ in range(warmup):
        _ = model.run(img_rgb)
    
    # Reset memory stats after warmup
    if track_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        result = model.run(img_rgb)
        
        if track_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    # Get peak memory usage
    peak_memory_mb = 0
    if track_memory and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'num_keypoints': len(result['keypoints']),
        'peak_memory_mb': peak_memory_mb
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark PyTorch vs ONNX Runtime (speed and memory)')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'],
                        default='aliked-n16rot', help='Model configuration')
    parser.add_argument('--onnx-path', type=str, default=None,
                        help='Path to ONNX model (default: models/{model}.onnx)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device: cpu or cuda (default: cuda if available)')
    parser.add_argument('--size', type=str, default='720x1280',
                        help='Test image size as HxW (default: 720x1280)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to real image for testing (default: use random image)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of benchmark runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup runs (default: 3)')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for benchmark results (default: benchmark_results.json)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Check CUDA availability
    track_memory = (args.device == 'cuda' and torch.cuda.is_available())
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        args.device = 'cpu'
    
    # Parse image size
    h, w = map(int, args.size.split('x'))
    
    # Load or create test image
    if args.image:
        if not os.path.exists(args.image):
            logging.error(f"Image not found: {args.image}")
            return
        
        logging.info(f"Loading test image: {args.image}")
        img = cv2.imread(args.image)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        if img_rgb.shape[:2] != (h, w):
            img_rgb = cv2.resize(img_rgb, (w, h))
            logging.info(f"Resized image to {h}x{w}")
    else:
        logging.info(f"Creating random test image: {h}x{w}")
        img_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Set ONNX path
    if args.onnx_path is None:
        args.onnx_path = f'models/{args.model}.onnx'
    
    print("\n" + "="*80)
    print(f"PyTorch vs ONNX Runtime Benchmark - {args.model}")
    print("="*80)
    print(f"Image size: {h}x{w}")
    print(f"Device: {args.device}")
    print(f"Memory tracking: {'Enabled' if track_memory else 'Disabled'}")
    print(f"Runs: {args.runs} (warmup: {args.warmup})")
    print("="*80 + "\n")
    
    # Benchmark PyTorch
    try:
        logging.info("Loading PyTorch model...")
        pytorch_model = ALIKED(
            model_name=args.model,
            device=args.device,
            top_k=-1,
            scores_th=0.2,
            n_limit=5000
        )
        
        logging.info("Benchmarking PyTorch...")
        pytorch_stats = benchmark_model(pytorch_model, img_rgb, args.runs, args.warmup, track_memory)
        
        print("PyTorch Results:")
        print(f"  Mean time:      {pytorch_stats['mean']*1000:.2f} ms")
        print(f"  Std dev:        {pytorch_stats['std']*1000:.2f} ms")
        print(f"  Min time:       {pytorch_stats['min']*1000:.2f} ms")
        print(f"  Max time:       {pytorch_stats['max']*1000:.2f} ms")
        if track_memory:
            print(f"  Peak memory:    {pytorch_stats['peak_memory_mb']:.2f} MB")
        print(f"  Keypoints:      {pytorch_stats['num_keypoints']}")
        print()
        
    except Exception as e:
        logging.error(f"PyTorch benchmark failed: {e}")
        pytorch_stats = None
        print("PyTorch: FAILED\n")
    
    # Benchmark ONNX
    try:
        if not os.path.exists(args.onnx_path):
            logging.error(f"ONNX model not found: {args.onnx_path}")
            logging.error(f"Export it first: python export_onnx.py --model {args.model}")
            return
        
        logging.info("Loading ONNX model...")
        onnx_model = ALIKED_ONNX(
            onnx_path=args.onnx_path,
            device=args.device,
            n_limit=5000
        )
        
        logging.info("Benchmarking ONNX Runtime...")
        onnx_stats = benchmark_model(onnx_model, img_rgb, args.runs, args.warmup, track_memory)
        
        print("ONNX Runtime Results:")
        print(f"  Mean time:      {onnx_stats['mean']*1000:.2f} ms")
        print(f"  Std dev:        {onnx_stats['std']*1000:.2f} ms")
        print(f"  Min time:       {onnx_stats['min']*1000:.2f} ms")
        print(f"  Max time:       {onnx_stats['max']*1000:.2f} ms")
        if track_memory:
            print(f"  Peak memory:    {onnx_stats['peak_memory_mb']:.2f} MB")
        print(f"  Keypoints:      {onnx_stats['num_keypoints']}")
        print()
        
    except Exception as e:
        logging.error(f"ONNX benchmark failed: {e}")
        onnx_stats = None
        print("ONNX: FAILED\n")
    
    # Compare
    if pytorch_stats and onnx_stats:
        print("="*80)
        print("Comparison:")
        print("="*80)
        
        # Speed comparison
        speedup = pytorch_stats['mean'] / onnx_stats['mean']
        if speedup > 1:
            print(f"Speed: ONNX is {speedup:.2f}x FASTER than PyTorch")
        else:
            print(f"Speed: PyTorch is {1/speedup:.2f}x FASTER than ONNX")
        
        print(f"       Absolute difference: {abs(pytorch_stats['mean'] - onnx_stats['mean'])*1000:.2f} ms")
        
        # Memory comparison
        if track_memory:
            mem_ratio = pytorch_stats['peak_memory_mb'] / onnx_stats['peak_memory_mb']
            if mem_ratio > 1:
                print(f"\nMemory: ONNX uses {mem_ratio:.2f}x LESS memory than PyTorch")
            else:
                print(f"\nMemory: PyTorch uses {1/mem_ratio:.2f}x LESS memory than ONNX")
            
            mem_diff = abs(pytorch_stats['peak_memory_mb'] - onnx_stats['peak_memory_mb'])
            print(f"        Absolute difference: {mem_diff:.2f} MB")
            print(f"        PyTorch: {pytorch_stats['peak_memory_mb']:.2f} MB")
            print(f"        ONNX:    {onnx_stats['peak_memory_mb']:.2f} MB")
        
        # Keypoint comparison
        kp_diff = abs(pytorch_stats['num_keypoints'] - onnx_stats['num_keypoints'])
        print(f"\nKeypoints: Difference of {kp_diff} ({kp_diff/pytorch_stats['num_keypoints']*100:.2f}%)")
        
        print("="*80 + "\n")
    
    # Save results to file
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model': args.model,
            'device': args.device,
            'image_size': f"{h}x{w}",
            'image_path': args.image if args.image else 'random',
            'runs': args.runs,
            'warmup': args.warmup,
            'memory_tracking': track_memory
        },
        'pytorch': pytorch_stats if pytorch_stats else None,
        'onnx': onnx_stats if onnx_stats else None,
        'comparison': None
    }
    
    if pytorch_stats and onnx_stats:
        results['comparison'] = {
            'speed_ratio': pytorch_stats['mean'] / onnx_stats['mean'],
            'time_difference_ms': abs(pytorch_stats['mean'] - onnx_stats['mean']) * 1000,
            'faster': 'onnx' if pytorch_stats['mean'] > onnx_stats['mean'] else 'pytorch',
            'keypoint_difference': abs(pytorch_stats['num_keypoints'] - onnx_stats['num_keypoints']),
            'keypoint_difference_pct': abs(pytorch_stats['num_keypoints'] - onnx_stats['num_keypoints']) / pytorch_stats['num_keypoints'] * 100
        }
        
        if track_memory:
            results['comparison']['memory_ratio'] = pytorch_stats['peak_memory_mb'] / onnx_stats['peak_memory_mb']
            results['comparison']['memory_difference_mb'] = abs(pytorch_stats['peak_memory_mb'] - onnx_stats['peak_memory_mb'])
            results['comparison']['less_memory'] = 'onnx' if pytorch_stats['peak_memory_mb'] > onnx_stats['peak_memory_mb'] else 'pytorch'
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
