"""
Convert benchmark JSON files to markdown format
"""

import json
import glob
import argparse
from datetime import datetime


def format_time_ms(seconds):
    """Convert seconds to milliseconds with 2 decimal places"""
    return f"{seconds * 1000:.2f} ms"


def format_memory_mb(mb):
    """Format memory in MB with 2 decimal places"""
    return f"{mb:.2f} MB"


def convert_json_to_md(json_files, output_file):
    """Convert multiple JSON benchmark files to a single markdown file"""
    
    with open(output_file, 'w') as md:
        md.write("# ALIKED Benchmark Results\n\n")
        md.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md.write("---\n\n")
        
        for json_file in sorted(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Header
            config = data['config']
            md.write(f"## Benchmark: {config['model']} on {config['device'].upper()}\n\n")
            
            # Configuration table
            md.write("### Configuration\n\n")
            md.write("| Parameter | Value |\n")
            md.write("|-----------|-------|\n")
            md.write(f"| Model | {config['model']} |\n")
            md.write(f"| Device | {config['device']} |\n")
            md.write(f"| Image Size | {config['image_size']} |\n")
            md.write(f"| Image Source | {config['image_path']} |\n")
            md.write(f"| Benchmark Runs | {config['runs']} |\n")
            md.write(f"| Warmup Runs | {config['warmup']} |\n")
            md.write(f"| Memory Tracking | {'✅ Enabled' if config['memory_tracking'] else '❌ Disabled'} |\n")
            md.write(f"| Timestamp | {data['timestamp']} |\n\n")
            
            # Results comparison table
            md.write("### Performance Results\n\n")
            md.write("| Metric | PyTorch | ONNX Runtime |\n")
            md.write("|--------|---------|-------------|\n")
            
            if data['pytorch']:
                pytorch = data['pytorch']
                onnx = data['onnx']
                
                md.write(f"| **Mean Time** | {format_time_ms(pytorch['mean'])} | {format_time_ms(onnx['mean'])} |\n")
                md.write(f"| Std Deviation | {format_time_ms(pytorch['std'])} | {format_time_ms(onnx['std'])} |\n")
                md.write(f"| Min Time | {format_time_ms(pytorch['min'])} | {format_time_ms(onnx['min'])} |\n")
                md.write(f"| Max Time | {format_time_ms(pytorch['max'])} | {format_time_ms(onnx['max'])} |\n")
                
                if config['memory_tracking']:
                    md.write(f"| **Peak Memory** | {format_memory_mb(pytorch['peak_memory_mb'])} | {format_memory_mb(onnx['peak_memory_mb'])} |\n")
                
                md.write(f"| **Keypoints Detected** | {pytorch['num_keypoints']} | {onnx['num_keypoints']} |\n\n")
                
                # Comparison summary
                if data['comparison']:
                    comp = data['comparison']
                    md.write("### Comparison Summary\n\n")
                    
                    # Speed comparison
                    if comp['faster'] == 'onnx':
                        md.write(f"**Speed:** 🚀 ONNX is **{comp['speed_ratio']:.2f}x faster** than PyTorch\n\n")
                    else:
                        md.write(f"**Speed:** 🚀 PyTorch is **{1/comp['speed_ratio']:.2f}x faster** than ONNX\n\n")
                    
                    md.write(f"- Time difference: {comp['time_difference_ms']:.2f} ms\n")
                    md.write(f"- PyTorch: {format_time_ms(pytorch['mean'])}\n")
                    md.write(f"- ONNX: {format_time_ms(onnx['mean'])}\n\n")
                    
                    # Memory comparison (if available)
                    if 'memory_ratio' in comp:
                        if comp['less_memory'] == 'onnx':
                            md.write(f"**Memory:** 💾 ONNX uses **{comp['memory_ratio']:.2f}x less** memory than PyTorch\n\n")
                        else:
                            md.write(f"**Memory:** 💾 PyTorch uses **{1/comp['memory_ratio']:.2f}x less** memory than ONNX\n\n")
                        
                        md.write(f"- Memory difference: {format_memory_mb(comp['memory_difference_mb'])}\n")
                        md.write(f"- PyTorch: {format_memory_mb(pytorch['peak_memory_mb'])}\n")
                        md.write(f"- ONNX: {format_memory_mb(onnx['peak_memory_mb'])}\n\n")
                    
                    # Keypoints comparison
                    if comp['keypoint_difference'] > 0:
                        md.write(f"**Keypoints:** Difference of {comp['keypoint_difference']} keypoints ({comp['keypoint_difference_pct']:.2f}%)\n\n")
                    else:
                        md.write(f"**Keypoints:** ✅ Same number of keypoints detected ({pytorch['num_keypoints']})\n\n")
            else:
                md.write("| *No results available* | - | - |\n\n")
            
            md.write("---\n\n")
        
        # Summary section
        md.write("## Summary\n\n")
        md.write("### Key Findings\n\n")
        
        # Analyze all benchmarks
        cpu_results = []
        cuda_results = []
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if data['comparison']:
                if data['config']['device'] == 'cpu':
                    cpu_results.append(data)
                elif data['config']['device'] == 'cuda':
                    cuda_results.append(data)
        
        if cpu_results:
            md.write("#### CPU Performance\n\n")
            for result in cpu_results:
                comp = result['comparison']
                faster = "ONNX" if comp['faster'] == 'onnx' else "PyTorch"
                ratio = comp['speed_ratio'] if comp['faster'] == 'onnx' else 1/comp['speed_ratio']
                md.write(f"- **{result['config']['image_size']}**: {faster} is {ratio:.2f}x faster\n")
            md.write("\n")
        
        if cuda_results:
            md.write("#### CUDA Performance\n\n")
            for result in cuda_results:
                comp = result['comparison']
                faster = "ONNX" if comp['faster'] == 'onnx' else "PyTorch"
                ratio = comp['speed_ratio'] if comp['faster'] == 'onnx' else 1/comp['speed_ratio']
                md.write(f"- **{result['config']['image_size']}**: {faster} is {ratio:.2f}x faster")
                
                if 'memory_ratio' in comp:
                    less_mem = "ONNX" if comp['less_memory'] == 'onnx' else "PyTorch"
                    mem_ratio = comp['memory_ratio'] if comp['less_memory'] == 'onnx' else 1/comp['memory_ratio']
                    md.write(f" | {less_mem} uses {mem_ratio:.2f}x less memory")
                
                md.write("\n")
            md.write("\n")
        
        md.write("### Recommendations\n\n")
        md.write("- **For CPU deployment**: ONNX Runtime offers better performance\n")
        md.write("- **For GPU deployment**: PyTorch with CUDA is significantly faster\n")
        md.write("- **For memory-constrained environments**: ONNX Runtime uses much less GPU memory\n")
        md.write("- **For production**: Consider ONNX Runtime on CPU for better throughput\n\n")


def main():
    parser = argparse.ArgumentParser(description='Convert benchmark JSON files to markdown')
    parser.add_argument('--input', type=str, default='benchmark_*.json',
                        help='Input JSON file pattern (default: benchmark_*.json)')
    parser.add_argument('--output', type=str, default='BENCHMARK_RESULTS.md',
                        help='Output markdown file (default: BENCHMARK_RESULTS.md)')
    
    args = parser.parse_args()
    
    # Find all matching JSON files
    json_files = glob.glob(args.input)
    
    if not json_files:
        print(f"No JSON files found matching pattern: {args.input}")
        return
    
    print(f"Found {len(json_files)} benchmark file(s):")
    for f in json_files:
        print(f"  - {f}")
    
    print(f"\nConverting to markdown: {args.output}")
    convert_json_to_md(json_files, args.output)
    print(f"✅ Successfully created {args.output}")


if __name__ == '__main__':
    main()
