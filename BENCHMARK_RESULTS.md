# ALIKED Benchmark Results

Generated on: 2026-05-17 16:18:01

---

## Benchmark: aliked-n16rot on CPU

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | aliked-n16rot |
| Device | cpu |
| Image Size | 480x640 |
| Image Source | assets/south_building/P1180143.jpg |
| Benchmark Runs | 3 |
| Warmup Runs | 3 |
| Memory Tracking | ❌ Disabled |
| Timestamp | 2026-05-17T16:15:20.312806 |

### Performance Results

| Metric | PyTorch | ONNX Runtime |
|--------|---------|-------------|
| **Mean Time** | 346.14 ms | 287.82 ms |
| Std Deviation | 4.35 ms | 2.03 ms |
| Min Time | 342.75 ms | 285.46 ms |
| Max Time | 352.27 ms | 290.41 ms |
| **Keypoints Detected** | 1603 | 5000 |

### Comparison Summary

**Speed:** 🚀 ONNX is **1.20x faster** than PyTorch

- Time difference: 58.31 ms
- PyTorch: 346.14 ms
- ONNX: 287.82 ms

**Keypoints:** Difference of 3397 keypoints (211.92%)

---

## Benchmark: aliked-n16rot on CUDA

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | aliked-n16rot |
| Device | cuda |
| Image Size | 720x1280 |
| Image Source | assets/south_building/P1180143.jpg |
| Benchmark Runs | 3 |
| Warmup Runs | 3 |
| Memory Tracking | ✅ Enabled |
| Timestamp | 2026-05-17T16:16:07.047809 |

### Performance Results

| Metric | PyTorch | ONNX Runtime |
|--------|---------|-------------|
| **Mean Time** | 49.91 ms | 715.87 ms |
| Std Deviation | 3.48 ms | 11.66 ms |
| Min Time | 47.42 ms | 706.59 ms |
| Max Time | 54.83 ms | 732.32 ms |
| **Peak Memory** | 1454.40 MB | 10.73 MB |
| **Keypoints Detected** | 5000 | 5000 |

### Comparison Summary

**Speed:** 🚀 PyTorch is **14.34x faster** than ONNX

- Time difference: 665.96 ms
- PyTorch: 49.91 ms
- ONNX: 715.87 ms

**Memory:** 💾 ONNX uses **135.55x less** memory than PyTorch

- Memory difference: 1443.67 MB
- PyTorch: 1454.40 MB
- ONNX: 10.73 MB

**Keypoints:** ✅ Same number of keypoints detected (5000)

---

## Benchmark: aliked-n16rot on CUDA

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | aliked-n16rot |
| Device | cuda |
| Image Size | 480x640 |
| Image Source | assets/south_building/P1180143.jpg |
| Benchmark Runs | 5 |
| Warmup Runs | 3 |
| Memory Tracking | ✅ Enabled |
| Timestamp | 2026-05-17T16:16:41.412033 |

### Performance Results

| Metric | PyTorch | ONNX Runtime |
|--------|---------|-------------|
| **Mean Time** | 15.18 ms | 256.32 ms |
| Std Deviation | 1.64 ms | 2.65 ms |
| Min Time | 13.91 ms | 253.63 ms |
| Max Time | 18.35 ms | 260.20 ms |
| **Peak Memory** | 482.26 MB | 10.73 MB |
| **Keypoints Detected** | 1603 | 5000 |

### Comparison Summary

**Speed:** 🚀 PyTorch is **16.88x faster** than ONNX

- Time difference: 241.14 ms
- PyTorch: 15.18 ms
- ONNX: 256.32 ms

**Memory:** 💾 ONNX uses **44.94x less** memory than PyTorch

- Memory difference: 471.53 MB
- PyTorch: 482.26 MB
- ONNX: 10.73 MB

**Keypoints:** Difference of 3397 keypoints (211.92%)

---

## Summary

### Key Findings

#### CPU Performance

- **480x640**: ONNX is 1.20x faster

#### CUDA Performance

- **480x640**: PyTorch is 16.88x faster | ONNX uses 44.94x less memory
- **720x1280**: PyTorch is 14.34x faster | ONNX uses 135.55x less memory

### Recommendations

- **For CPU deployment**: ONNX Runtime offers better performance
- **For GPU deployment**: PyTorch with CUDA is significantly faster
- **For memory-constrained environments**: ONNX Runtime uses much less GPU memory
- **For production**: Consider ONNX Runtime on CPU for better throughput

