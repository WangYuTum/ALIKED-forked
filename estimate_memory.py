"""
CUDA Memory Estimation Tool for ALIKED
This script helps estimate CUDA memory usage for different image sizes and model configurations.
"""

import torch
import numpy as np
from nets.aliked import ALIKED
import argparse
import logging


def get_memory_stats():
    """Get current CUDA memory statistics in MB"""
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    return allocated, reserved, max_allocated


def estimate_memory_usage(model_name='aliked-n16rot', device='cuda', image_sizes=None, n_limit=5000):
    """
    Estimate CUDA memory usage for different image sizes
    
    Args:
        model_name: Model configuration name
        device: Device to run on
        image_sizes: List of (H, W) tuples to test. If None, uses default sizes.
        n_limit: Maximum number of keypoints to detect
    
    Returns:
        Dictionary with memory usage information for each image size
    """
    
    if image_sizes is None:
        # Common image sizes to test
        image_sizes = [
            (480, 640),    # VGA
            (600, 800),    # SVGA
            (720, 1280),   # HD
            (1080, 1920),  # Full HD
            (1536, 2048),  # 2K
        ]
    
    if not torch.cuda.is_available():
        logging.error("CUDA not available!")
        return None
    
    results = {}
    
    # Initialize model
    logging.info(f"Loading model: {model_name} with n_limit={n_limit}")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    initial_allocated, initial_reserved, _ = get_memory_stats()
    
    model = ALIKED(model_name=model_name,
                   device=device,
                   top_k=-1,
                   scores_th=0.2,
                   n_limit=n_limit)
    
    model_allocated, model_reserved, _ = get_memory_stats()
    model_memory = model_allocated - initial_allocated
    
    logging.info(f"Model parameters memory: {model_memory:.2f} MB")
    
    # Test each image size
    for h, w in image_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Create dummy RGB image
        img_rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        
        # Measure memory before inference
        before_alloc, before_reserved, _ = get_memory_stats()
        
        # Run inference with error handling for OOM
        try:
            with torch.no_grad():
                pred = model.run(img_rgb)
        except torch.cuda.OutOfMemoryError as e:
            logging.warning(f"OOM for size {h}x{w}: {str(e)}")
            # Clean up and continue
            torch.cuda.empty_cache()
            results[(h, w)] = {
                'image_size': f"{h}x{w}",
                'padded_size': f"{((h-1)//32+1)*32}x{((w-1)//32+1)*32}",
                'num_keypoints': -1,
                'model_memory_mb': model_memory,
                'inference_memory_mb': -1,
                'peak_memory_mb': -1,
                'total_memory_mb': -1,
                'estimated_feature_memory_mb': -1,
                'inference_time_sec': -1,
                'oom': True,
            }
            continue
        
        # Measure memory after inference
        after_alloc, after_reserved, max_alloc = get_memory_stats()
        
        # Calculate memory usage
        inference_memory = after_alloc - before_alloc
        peak_memory = max_alloc - initial_allocated
        total_memory = max_alloc
        
        # Calculate feature map sizes (approximate)
        # Images are padded to be divisible by 32
        padded_h = ((h - 1) // 32 + 1) * 32
        padded_w = ((w - 1) // 32 + 1) * 32
        
        # Get model configuration
        from nets.aliked import ALIKED_CFGS
        cfg = ALIKED_CFGS[model_name]
        
        # Estimate feature map memory (in MB)
        # Feature maps: x1(H×W), x2(H/2×W/2), x3(H/8×W/8), x4(H/32×W/32), x1234(H×W)
        feature_memory = (
            padded_h * padded_w * cfg['c1'] +  # x1
            (padded_h//2) * (padded_w//2) * cfg['c2'] +  # x2
            (padded_h//8) * (padded_w//8) * cfg['c3'] +  # x3
            (padded_h//32) * (padded_w//32) * cfg['c4'] +  # x4
            padded_h * padded_w * cfg['dim']  # x1234 (aggregated features)
        ) * 4 / 1024**2  # 4 bytes per float32, convert to MB
        
        results[(h, w)] = {
            'image_size': f"{h}x{w}",
            'padded_size': f"{padded_h}x{padded_w}",
            'num_keypoints': len(pred['keypoints']),
            'model_memory_mb': model_memory,
            'inference_memory_mb': inference_memory,
            'peak_memory_mb': peak_memory,
            'total_memory_mb': total_memory,
            'estimated_feature_memory_mb': feature_memory,
            'inference_time_sec': pred['time'],
        }
        
        logging.info(f"Size {h}x{w}: Peak={peak_memory:.2f}MB, Total={total_memory:.2f}MB, "
                     f"Keypoints={len(pred['keypoints'])}, Time={pred['time']:.3f}s")
    
    return results


def print_memory_report(results, model_name):
    """Print a formatted memory usage report"""
    print("\n" + "="*80)
    print(f"CUDA Memory Usage Report for {model_name}")
    print("="*80)
    
    print(f"\n{'Image Size':<15} {'Padded':<15} {'Keypoints':<10} {'Model':<10} {'Peak':<10} {'Total':<10} {'Time(s)':<10}")
    print(f"{'(HxW)':<15} {'Size':<15} {'Count':<10} {'(MB)':<10} {'(MB)':<10} {'(MB)':<10} {'':<10}")
    print("-"*80)
    
    for size_key in sorted(results.keys()):
        r = results[size_key]
        if r.get('oom', False):
            print(f"{r['image_size']:<15} {r['padded_size']:<15} {'OOM':<10} "
                  f"{r['model_memory_mb']:<10.2f} {'OOM':<10} {'OOM':<10} {'N/A':<10}")
        else:
            print(f"{r['image_size']:<15} {r['padded_size']:<15} {r['num_keypoints']:<10} "
                  f"{r['model_memory_mb']:<10.2f} {r['peak_memory_mb']:<10.2f} "
                  f"{r['total_memory_mb']:<10.2f} {r['inference_time_sec']:<10.3f}")
    
    print("\n" + "="*80)
    print("Memory Components:")
    print("  - Model: Memory occupied by model parameters (weights, biases)")
    print("  - Peak: Maximum memory used during inference (includes activations)")
    print("  - Total: Total CUDA memory allocated on device")
    print("  - OOM: Out of Memory - insufficient GPU memory for this image size")
    print("\nNote: Peak memory = Model parameters + Feature maps + Activations + Keypoint data")
    print("="*80 + "\n")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate CUDA memory usage for ALIKED models')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'], 
                        default='aliked-n16rot', help='Model configuration')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Custom image sizes to test, format: "480x640,600x800,720x1280,1080x1920"')
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to detect (default: 5000)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse custom sizes if provided
    image_sizes = None
    if args.sizes:
        image_sizes = []
        for size_str in args.sizes.split(','):
            h, w = map(int, size_str.split('x'))
            image_sizes.append((h, w))
    
    # Run actual memory profiling
    results = estimate_memory_usage(args.model, args.device, image_sizes, args.n_limit)
    
    if results:
        print_memory_report(results, args.model)
