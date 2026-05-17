"""
Benchmark PyTorch vs ONNX Runtime performance
"""

import time
import numpy as np
import argparse
import logging
from nets.aliked import ALIKED
from nets.aliked_onnx import ALIKED_ONNX


def benchmark_model(model, img_rgb, num_runs=10, warmup=3):
    """
    Benchmark model inference time
    
    Args:
        model: Model instance (PyTorch or ONNX)
        img_rgb: RGB image
        num_runs: Number of inference runs
        warmup: Number of warmup runs
    
    Returns:
        Average inference time in seconds
    """
    # Warmup
    for _ in range(warmup):
        _ = model.run(img_rgb)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        result = model.run(img_rgb)
        end = time.time()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'num_keypoints': len(result['keypoints'])
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark PyTorch vs ONNX Runtime')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'],
                        default='aliked-n16rot', help='Model configuration')
    parser.add_argument('--onnx-path', type=str, default=None,
                        help='Path to ONNX model (default: models/{model}.onnx)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--size', type=str, default='720x1280',
                        help='Test image size as HxW (default: 720x1280)')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of benchmark runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup runs (default: 3)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse image size
    h, w = map(int, args.size.split('x'))
    
    # Create dummy image
    logging.info(f"Creating test image: {h}x{w}")
    img_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    # Set ONNX path
    if args.onnx_path is None:
        args.onnx_path = f'models/{args.model}.onnx'
    
    print("\n" + "="*80)
    print(f"PyTorch vs ONNX Runtime Benchmark - {args.model}")
    print("="*80)
    print(f"Image size: {h}x{w}")
    print(f"Device: {args.device}")
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
        pytorch_stats = benchmark_model(pytorch_model, img_rgb, args.runs, args.warmup)
        
        print("PyTorch Results:")
        print(f"  Mean time:      {pytorch_stats['mean']*1000:.2f} ms")
        print(f"  Std dev:        {pytorch_stats['std']*1000:.2f} ms")
        print(f"  Min time:       {pytorch_stats['min']*1000:.2f} ms")
        print(f"  Max time:       {pytorch_stats['max']*1000:.2f} ms")
        print(f"  Keypoints:      {pytorch_stats['num_keypoints']}")
        print()
        
    except Exception as e:
        logging.error(f"PyTorch benchmark failed: {e}")
        pytorch_stats = None
        print("PyTorch: FAILED\n")
    
    # Benchmark ONNX
    try:
        import os
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
        onnx_stats = benchmark_model(onnx_model, img_rgb, args.runs, args.warmup)
        
        print("ONNX Runtime Results:")
        print(f"  Mean time:      {onnx_stats['mean']*1000:.2f} ms")
        print(f"  Std dev:        {onnx_stats['std']*1000:.2f} ms")
        print(f"  Min time:       {onnx_stats['min']*1000:.2f} ms")
        print(f"  Max time:       {onnx_stats['max']*1000:.2f} ms")
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
        
        speedup = pytorch_stats['mean'] / onnx_stats['mean']
        if speedup > 1:
            print(f"ONNX is {speedup:.2f}x FASTER than PyTorch")
        else:
            print(f"PyTorch is {1/speedup:.2f}x FASTER than ONNX")
        
        print(f"Absolute difference: {abs(pytorch_stats['mean'] - onnx_stats['mean'])*1000:.2f} ms")
        
        kp_diff = abs(pytorch_stats['num_keypoints'] - onnx_stats['num_keypoints'])
        print(f"Keypoint difference: {kp_diff} ({kp_diff/pytorch_stats['num_keypoints']*100:.2f}%)")
        
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
