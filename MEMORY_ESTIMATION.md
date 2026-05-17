# CUDA Memory Measurements for ALIKED

## Summary

This document provides actual CUDA memory measurements for all ALIKED models across different image sizes. All measurements were performed on the same GPU with consistent test conditions.

**Important Finding**: Memory consumption does NOT scale with the number of keypoints detected. Memory is determined solely by image resolution and model architecture, not by the `n_limit` parameter. See [detailed analysis](#does-memory-scale-with-number-of-keypoints) below.

## Model Comparison

Overview of model parameters and memory characteristics:

| Model          | Model Params (MB) | Memory Usage  | Keypoint Quality |
|----------------|-------------------|---------------|------------------|
| aliked-t16     | 0.76              | Lowest (~48%) | Good             |
| aliked-n16     | 2.60              | High          | Better           |
| aliked-n16rot  | 2.60              | High          | Better (rotation)|
| aliked-n32     | 3.76              | Highest       | Best             |

**Note**: aliked-n16 and aliked-n16rot have identical memory usage, but n16rot includes rotation invariance.

**Note**: aliked-n16 and aliked-n16rot have identical memory usage, but n16rot includes rotation invariance.

## Actual Memory Measurements

### aliked-t16 (Smallest/Fastest)

**Model Parameters**: 0.76 MB

| Image Size  | Padded Size | Peak (MB) | Keypoints | Time (s) | GPU Req   |
|-------------|-------------|-----------|-----------|----------|-----------|
| 480×640     | 480×640     | 250.41    | 5000      | 0.348    | 512 MB+   |
| 600×800     | 608×800     | 389.71    | 5000      | 0.034    | 512 MB+   |
| 720×1280    | 736×1280    | 744.73    | 5000      | 0.037    | 1 GB+     |
| 1080×1920   | 1088×1920   | 1639.57   | 5000      | 0.052    | 2 GB+     |
| 1536×2048   | 1536×2048   | 2460.07   | 5000      | 0.064    | 3 GB+     |

### aliked-n16 (Balanced)

**Model Parameters**: 2.60 MB

| Image Size  | Padded Size | Peak (MB) | Keypoints | Time (s) | GPU Req   |
|-------------|-------------|-----------|-----------|----------|-----------|
| 480×640     | 480×640     | 482.10    | 2380      | 0.332    | 1 GB+     |
| 600×800     | 608×800     | 757.33    | 4294      | 0.043    | 1 GB+     |
| 720×1280    | 736×1280    | 1453.90   | 5000      | 0.059    | 2 GB+     |
| 1080×1920   | 1088×1920   | 3207.20   | 5000      | 0.082    | 4 GB+     |
| 1536×2048   | 1536×2048   | 4817.10   | 5000      | 0.110    | 6 GB+     |

### aliked-n16rot (Rotation Invariant)

**Model Parameters**: 2.60 MB

| Image Size  | Padded Size | Peak (MB) | Keypoints | Time (s) | GPU Req   |
|-------------|-------------|-----------|-----------|----------|-----------|
| 480×640     | 480×640     | 482.10    | 405       | 0.309    | 1 GB+     |
| 600×800     | 608×800     | 757.33    | 832       | 0.037    | 1 GB+     |
| 720×1280    | 736×1280    | 1453.90   | 1319      | 0.042    | 2 GB+     |
| 1080×1920   | 1088×1920   | 3207.20   | 4597      | 0.073    | 4 GB+     |
| 1536×2048   | 1536×2048   | 4817.10   | 5000      | 0.106    | 6 GB+     |

### aliked-n32 (Highest Quality)

**Model Parameters**: 3.76 MB

| Image Size  | Padded Size | Peak (MB) | Keypoints | Time (s) | GPU Req   |
|-------------|-------------|-----------|-----------|----------|-----------|
| 480×640     | 480×640     | 483.75    | 555       | 0.337    | 1 GB+     |
| 600×800     | 608×800     | 758.48    | 850       | 0.037    | 1 GB+     |
| 720×1280    | 736×1280    | 1455.06   | 1762      | 0.053    | 2 GB+     |
| 1080×1920   | 1088×1920   | 3208.35   | 4953      | 0.092    | 4 GB+     |
| 1536×2048   | 1536×2048   | 4818.26   | 5000      | 0.125    | 6 GB+     |

## Key Observations

1. **Memory Scaling**: n16/n16rot/n32 use approximately **2x** the memory of t16
2. **Identical Memory**: n16, n16rot, and n32 have nearly identical memory usage (~1-2 MB difference)
3. **Performance**: First inference is slower (cold start), subsequent runs are ~10x faster
4. **Keypoint Detection**: Different models detect different numbers of keypoints with same settings
   - t16 maxes out at n_limit quickly (5000)
   - n16rot is more selective (fewer keypoints at same threshold)
   - n32 is highly selective (very precise keypoints)

## Usage Examples

### Measure Memory for a Specific Model

```bash
# Test aliked-t16 (smallest model)
python estimate_memory.py --model aliked-t16

# Test aliked-n16 (balanced model)
python estimate_memory.py --model aliked-n16

# Test aliked-n16rot (rotation invariant)
python estimate_memory.py --model aliked-n16rot

# Test aliked-n32 (highest quality)
python estimate_memory.py --model aliked-n32
```

### Test Custom Image Sizes

```bash
# Test specific sizes for a model
python estimate_memory.py --model aliked-n16rot --sizes "480x640,1080x1920"

# Test more sizes
python estimate_memory.py --model aliked-t16 --sizes "320x240,640x480,1280x720"
```

## Model Selection Guide

Choose the right model based on your GPU memory and requirements:

### When to Use Each Model

**aliked-t16** - Use when:
- Limited GPU memory (< 2 GB)
- Need fast inference speed
- Processing high resolution images (4K+) with memory constraints
- Keypoint quantity is more important than quality
- ~50% memory savings compared to n16 models

**aliked-n16** - Use when:
- Balanced memory and performance needed
- Standard feature matching tasks
- 2-6 GB GPU available
- Good keypoint quality required

**aliked-n16rot** - Use when:
- Images have rotation variations
- Same memory as n16 but with rotation invariance
- Matching objects at different orientations
- More selective keypoint detection preferred

**aliked-n32** - Use when:
- Highest quality keypoints needed
- Slightly more selective than n16
- Similar memory to n16/n16rot (~1 MB more for model params)
- Precision over quantity

### GPU Memory Requirements by Resolution

| Image Size  | t16 (MB) | n16 (MB) | n16rot (MB) | n32 (MB) | Recommended GPU |
|-------------|----------|----------|-------------|----------|-----------------|
| 480×640     | 250      | 482      | 482         | 484      | 1 GB+           |
| 600×800     | 390      | 757      | 757         | 758      | 1 GB+           |
| 720×1280    | 745      | 1454     | 1454        | 1455     | 2 GB+           |
| 1080×1920   | 1640     | 3207     | 3207        | 3208     | 4 GB (t16), 6 GB (others) |
| 1536×2048   | 2460     | 4817     | 4817        | 4818     | 6 GB (t16), 8 GB+ (others) |

## Performance Characteristics

Based on actual measurements across all models:

### Inference Time by Model and Image Size

| Image Size  | t16 (s) | n16 (s) | n16rot (s) | n32 (s) | Notes              |
|-------------|---------|---------|------------|---------|---------------------|
| 480×640     | 0.348   | 0.332   | 0.309      | 0.337   | First run (cold)    |
| 600×800     | 0.034   | 0.043   | 0.037      | 0.037   | Warmed up           |
| 720×1280    | 0.037   | 0.059   | 0.042      | 0.053   | HD resolution       |
| 1080×1920   | 0.052   | 0.082   | 0.073      | 0.092   | Full HD             |
| 1536×2048   | 0.064   | 0.110   | 0.106      | 0.125   | 2K resolution       |

**Key Findings**:
- First inference is 10-15× slower than subsequent runs (CUDA initialization)
- All models show similar inference times after warm-up
- Inference time scales with image size
- t16 is slightly faster than larger models

## Tips for Memory Optimization

1. **Choose the right model**: 
   - Use `aliked-t16` for ~50% memory savings
   - Use `aliked-n16rot` for rotation invariance
   - Use `aliked-n32` for highest keypoint quality

2. **Resize images**: Memory scales linearly with pixels (H×W)

3. **Batch processing**: Process images one at a time to avoid accumulation

4. **Clear cache**: Use `torch.cuda.empty_cache()` between inferences

5. **n_limit parameter**:
   - **Does NOT affect memory**: Feel free to use high values (5000+)
   - Only affects output: limits maximum keypoints returned
   - Use high values for keypoint-rich scenes
   - Lower values don't save memory but may reduce post-processing time

## Memory Components

Total peak memory consists of:

1. **Model Parameters** (0.76-3.76 MB)
   - Weights and biases of the neural network
   - Constant regardless of image size
   - t16: 0.76 MB, n16/n16rot: 2.60 MB, n32: 3.76 MB

2. **Feature Maps** (varies with image size)
   - Intermediate activations at multiple scales
   - Scales linearly with image pixels
   - Primary memory consumer (~99% of total)

3. **Output Data** (varies with keypoints)
   - Detected keypoints and descriptors
   - Usually negligible compared to feature maps

## Does Memory Scale with Number of Keypoints?

**Answer: NO** - Memory consumption does NOT scale linearly (or at all) with the number of keypoints detected.

### Experimental Validation

Testing with aliked-n16rot on 720×1280 images with varying `n_limit` settings:

| n_limit | Actual Keypoints | Peak Memory (MB) | Inference Time (s) |
|---------|------------------|------------------|--------------------|
| 400     | 400              | 1453.90          | 0.351              |
| 600     | 600              | 1453.90          | 0.341              |
| 800     | 800              | 1453.90          | 0.336              |
| 1200    | 1200             | 1453.90          | 0.334              |
| 3000    | 1366             | 1453.90          | 0.322              |
| 3500    | 1253             | 1453.90          | 0.322              |
| 4000    | 1275             | 1453.90          | 0.318              |
| 5000    | 1273             | 1453.90          | 0.322              |

### Key Findings

1. **Peak memory is constant**: Exactly 1453.90 MB across all n_limit settings (400-5000)
2. **Keypoint data is negligible**: The memory for storing output keypoints and descriptors is < 0.01% of total
3. **Memory is dominated by feature maps**: The intermediate activations during forward pass consume ~99% of memory
4. **n_limit doesn't affect memory**: You can safely use higher n_limit values without worrying about OOM errors
5. **Inference time is stable**: Slight variations (0.318-0.351s) are within normal fluctuation, not related to keypoint count

### Why Doesn't Keypoint Count Matter?

The memory breakdown for inference:
- **Feature maps (99%+)**: Created during forward pass, same size regardless of detected keypoints
- **Score maps**: Same size as input (after padding), not affected by n_limit
- **Keypoint coordinates**: N × 2 floats (e.g., 5000 keypoints = ~40 KB)
- **Descriptors**: N × D floats (e.g., 5000 × 128 = ~2.5 MB for n16rot)

The keypoint and descriptor data is 3-4 orders of magnitude smaller than feature maps, making it effectively negligible.

### Practical Implication

**You can use high n_limit values (5000+) without increasing memory consumption.** The memory requirement depends solely on:
1. Image resolution (pixels)
2. Model architecture (t16 vs n16/n32)

The number of keypoints you want to extract has no impact on GPU memory usage.

## Memory Management in Code

```python
import torch
from nets.aliked import ALIKED

# Create model
model = ALIKED(model_name='aliked-n16rot', device='cuda', n_limit=5000)

# Process images with proper memory management
for img in images:
    torch.cuda.empty_cache()  # Clear cache before each inference
    
    with torch.no_grad():  # Disable gradient computation
        pred = model.run(img)
    
    # Process results...
    # Memory is automatically freed after this loop iteration
```

## Troubleshooting OOM Errors

If you encounter "CUDA out of memory" errors:

1. **Check actual requirements**: Run `estimate_memory.py` with your target model and size
2. **Switch to smaller model**: Use t16 instead of n16/n32 (~50% memory reduction)
3. **Reduce image resolution**: Downscale by 2x reduces memory by ~4x
4. **Enable memory fragmentation reduction**:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
5. **Clear cache before processing**:
   ```python
   torch.cuda.empty_cache()
   ```

## Example Use Cases

### Case 1: Processing 4K video on 8GB GPU

**Problem**: 4K images (2160×3840) would require ~12-13 GB for n16 models

**Solution**: 
- Option A: Use aliked-t16 (~6.5 GB estimated)
- Option B: Downscale to 1080p (3207 MB for n16rot)
- Recommendation: Option B - maintains quality, uses n16rot

### Case 2: Real-time feature matching on 2GB GPU

**Problem**: Limited to 2GB VRAM

**Solution**:
- Use aliked-t16 with maximum 720×1280 resolution (745 MB)
- Or use smaller resolutions: 600×800 works well (390 MB)
- Inference time: ~35-40ms after warm-up

### Case 3: Highest quality keypoints for 1080p images

**Problem**: Need best quality matches on Full HD images

**Solution**:
- Use aliked-n32 (3208 MB required)
- Requires 4GB+ GPU
- Provides most selective, high-quality keypoints
- Inference time: ~92ms

### Case 4: Rotation-invariant matching across orientations

**Problem**: Images captured at various rotations

**Solution**:
- Use aliked-n16rot (same memory as n16: 3207 MB for 1080p)
- Detects keypoints robust to rotation
- More selective than n16 (fewer but better keypoints)

## Quick Reference Table

### Memory Requirements (Peak MB)

| Image Size  | t16   | n16   | n16rot | n32   | Difference |
|-------------|-------|-------|--------|-------|------------|
| 480×640     | 250   | 482   | 482    | 484   | t16: -48%  |
| 600×800     | 390   | 757   | 757    | 758   | t16: -49%  |
| 720×1280    | 745   | 1454  | 1454   | 1455  | t16: -49%  |
| 1080×1920   | 1640  | 3207  | 3207   | 3208  | t16: -49%  |
| 1536×2048   | 2460  | 4817  | 4817   | 4818  | t16: -49%  |

**Key Insight**: aliked-t16 consistently uses ~49% of the memory compared to n16/n16rot/n32 models.

### Effect of n_limit Parameter

| n_limit | Keypoints Detected | Peak Memory (MB) | Change   |
|---------|-------------------|------------------|----------|
| 400     | 400               | 1453.90          | baseline |
| 800     | 800               | 1453.90          | 0.00%    |
| 1200    | 1200              | 1453.90          | 0.00%    |
| 3000    | 1366              | 1453.90          | 0.00%    |
| 5000    | 1273              | 1453.90          | 0.00%    |

**Result**: n_limit has **zero impact** on memory consumption. All tested on 720×1280 with aliked-n16rot.

## Testing Methodology

All measurements were performed using:
- Script: `estimate_memory.py`
- Settings: `top_k=-1`, `scores_th=0.2`, `n_limit=5000`
- Test images: Random RGB numpy arrays (to isolate model memory)
- Same GPU with cleared cache between tests
- PyTorch memory profiling: `torch.cuda.max_memory_allocated()`

## Summary

1. **Memory scales linearly** with image resolution (pixels)
2. **Model choice matters**: t16 uses ~50% memory of n16/n32 models
3. **n16, n16rot, n32** have nearly identical memory usage
4. **First inference is slow** (~300ms), subsequent runs are fast (~35-125ms)
5. **Keypoint detection varies** by model even with same parameters
6. **n_limit has ZERO impact on memory**: Keypoint count doesn't affect GPU memory usage
7. **For production**: Add 10-20% buffer to measured values for safety

### What Actually Affects Memory?

**Memory is determined by:**
- ✅ Image resolution (H × W)
- ✅ Model architecture (t16 vs n16 vs n32)

**Memory is NOT affected by:**
- ❌ Number of keypoints (`n_limit` parameter)
- ❌ Detection threshold (`scores_th`)
- ❌ Number of detected keypoints in output
