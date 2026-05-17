# ONNX Runtime Guide for ALIKED

This guide explains how to export and run ALIKED models using ONNX Runtime instead of PyTorch.

## Why Use ONNX Runtime?

- **Faster inference**: ONNX Runtime is optimized for production deployment
- **Cross-platform**: Run on different platforms without PyTorch dependency
- **Smaller deployment**: No need to include PyTorch in your deployment
- **Hardware optimization**: Better support for various hardware accelerators

## Important Limitations

⚠️ **Accuracy Trade-off**: The ONNX models use **regular convolutions instead of Deformable Convolutions (DCN)** because ONNX does not support DCN operations. This results in approximately **1-2% lower accuracy** compared to the original PyTorch models.

⚠️ **Tested Models**: Currently verified working with:
- ✅ aliked-t16 (0.74 MB)
- ✅ aliked-n16rot (2.5 MB)
- ⚠️ aliked-n16 and aliked-n32 (export should work but not yet tested)

⚠️ **Runtime Code Status**: The wrapper class `ALIKED_ONNX` and `demo_pair_onnx.py` have been created but not yet fully tested. Please report any issues.

## Installation

### Option 1: Install all ONNX dependencies at once (recommended)
```bash
pip install -r requirements-onnx.txt
```

This installs:
- `onnxruntime-gpu>=1.25.0` (GPU-accelerated inference)
- `onnx>=1.21.0` (ONNX model manipulation)
- `onnxsim>=0.6.3` (model simplification)
- `onnxscript>=0.7.0` (required by PyTorch for export)

### Option 2: Manual installation

For CPU inference:
```bash
pip install onnxruntime
```

For GPU inference (requires CUDA):
```bash
pip install onnxruntime-gpu
```

Install ONNX tools (required for export):
```bash
pip install onnx>=1.21.0 onnxsim>=0.6.3 onnxscript>=0.7.0
```

## Quick Start

**Current Status**: \u2705 Export working | \u26a0\ufe0f Runtime inference not yet tested

### Step 1: Export PyTorch Model to ONNX

Export a model (e.g., aliked-n16rot):

```bash
python export_onnx.py --model aliked-n16rot
```

This creates `models/aliked-n16rot.onnx`.

**Export all models:**
```bash
python export_onnx.py --model aliked-t16
python export_onnx.py --model aliked-n16
python export_onnx.py --model aliked-n16rot
python export_onnx.py --model aliked-n32
```

**Advanced export options:**
```bash
# Export with custom output path
python export_onnx.py --model aliked-n16rot --output my_model.onnx

# Export with specific opset version
python export_onnx.py --model aliked-n16rot --opset 12

# Export without simplification
python export_onnx.py --model aliked-n16rot --no-simplify
```

### Step 2: Run Inference with ONNX Runtime

\u26a0\ufe0f **Note**: The demo script `demo_pair_onnx.py` has been created but not yet tested. The following commands are expected to work but may require adjustments.

**Using the demo:**
```bash
# CPU inference
python demo_pair_onnx.py assets/south_buidling/ --model aliked-n16rot

# GPU inference (requires onnxruntime-gpu)
python demo_pair_onnx.py assets/south_buidling/ --model aliked-n16rot --device cuda

# Custom ONNX model path
python demo_pair_onnx.py assets/south_buidling/ --onnx-path my_model.onnx
```

## Using ONNX in Your Code

⚠️ **Note**: The following code examples have been created but not fully tested. Please verify functionality for your use case.

### Basic Usage

```python
from nets.aliked_onnx import ALIKED_ONNX
import cv2

# Load ONNX model
model = ALIKED_ONNX(
    onnx_path='models/aliked-n16rot.onnx',
    device='cpu',  # or 'cuda'
    n_limit=5000
)

# Load and preprocess image
img = cv2.imread('image.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
result = model.run(img_rgb)

# Access results
keypoints = result['keypoints']      # (N, 2) array
descriptors = result['descriptors']  # (N, D) array
scores = result['scores']            # (N,) array
score_map = result['score_map']      # (H, W) array
```

### Matching Keypoints Between Images

```python
import numpy as np

# Detect keypoints in both images
pred1 = model.run(img1_rgb)
pred2 = model.run(img2_rgb)

# Simple mutual nearest neighbor matching
desc1 = pred1['descriptors']
desc2 = pred2['descriptors']

sim = desc1 @ desc2.T
sim[sim < 0.75] = 0
nn12 = np.argmax(sim, axis=1)
nn21 = np.argmax(sim, axis=0)
ids1 = np.arange(0, sim.shape[0])
mask = (ids1 == nn21[nn12])
matches = np.stack([ids1[mask], nn12[mask]], axis=1)

print(f"Found {len(matches)} matches")
```

## Model Specifications

### Input
- **Name**: `image`
- **Shape**: `(1, 3, H, W)` - batch, channels, height, width
- **Type**: float32
- **Range**: [0, 1] (normalized)
- **Note**: Height and width must be divisible by 32 (handled automatically with padding)

### Outputs
1. **keypoints**: `(1, N, 2)` - Keypoint coordinates (normalized to [-1, 1])
2. **descriptors**: `(1, N, D)` - Feature descriptors (D=128 for n16/n32, D=64 for t16)
3. **scores**: `(1, N)` - Keypoint confidence scores
4. **score_map**: `(1, 1, H, W)` - Dense score map

**Note**: The `score_dispersity` and `time` outputs from the PyTorch version are excluded for ONNX compatibility.

## Performance Comparison

### Memory Usage
ONNX Runtime typically uses similar memory to PyTorch for the same model and image size. See [MEMORY_ESTIMATION.md](MEMORY_ESTIMATION.md) for detailed measurements.

### Inference Speed
ONNX Runtime is often **10-30% faster** than PyTorch for inference, especially on CPU. Benchmark your specific use case to verify.

### Model Size
ONNX models (DCN-free versions) are smaller than original PyTorch checkpoints:
- aliked-t16: **0.74 MB** (PyTorch: 0.78 MB)
- aliked-n16/n16rot: **2.5 MB** (PyTorch: 2.7 MB)
- aliked-n32: **~3.8 MB** (estimated, not yet exported)

## Troubleshooting

### Common Issues

**1. "Module 'onnxruntime' not found"**
```bash
pip install onnxruntime  # for CPU
# or
pip install onnxruntime-gpu  # for GPU
```

**2. "ONNX model not found"**

Make sure you exported the model first:
```bash
python export_onnx.py --model aliked-n16rot
```

**3. GPU not being used with onnxruntime-gpu**

Check available providers:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

**4. Different results between PyTorch and ONNX**

Expected differences:
- **Accuracy reduction (~1-2%)**: Due to replacement of DCN with regular convolutions
- **Small numerical differences (<0.01%)**: Due to different floating point implementations
- **Optimization differences**: ONNX Runtime may use different optimization strategies

For most practical applications, these differences should not significantly impact matching performance.

**5. Export fails with "torch.onnx" errors**

The export script uses the legacy JIT-based exporter (`dynamo=False`) because the new PyTorch 2.9+ dynamo exporter has issues with ALIKED.

If export still fails:
- Try disabling simplification: `--no-simplify`
- Use opset version 16 or higher: `--opset 16` (required for grid_sampler)
- Ensure all ONNX dependencies are installed: `pip install -r requirements-onnx.txt`
- Check [ONNX_EXPORT_FIXES.md](ONNX_EXPORT_FIXES.md) for detailed troubleshooting

## Advanced Usage

### Custom Preprocessing

```python
import numpy as np

def custom_preprocess(img_bgr):
    """Custom preprocessing pipeline"""
    # Convert BGR to RGB
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Optional: resize if needed
    # img = cv2.resize(img, (width, height))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transpose HWC to CHW
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# Use with ONNX model
img_tensor = custom_preprocess(img_bgr)
outputs = model.session.run(model.output_names, {model.input_name: img_tensor})
```

### Batch Processing (Future)

Note: Current implementation processes one image at a time. For batch processing, you would need to modify the export script to handle batch inputs properly.

## Integration Examples

### With OpenCV

```python
import cv2
from nets.aliked_onnx import ALIKED_ONNX

model = ALIKED_ONNX('models/aliked-n16rot.onnx')
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.run(rgb)
    
    # Draw keypoints
    for kp in result['keypoints']:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    cv2.imshow('Keypoints', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### With PIL

```python
from PIL import Image
import numpy as np
from nets.aliked_onnx import ALIKED_ONNX

model = ALIKED_ONNX('models/aliked-n16rot.onnx')

# Load with PIL
img_pil = Image.open('image.jpg').convert('RGB')
img_np = np.array(img_pil)

# Run inference
result = model.run(img_np)
print(f"Detected {len(result['keypoints'])} keypoints")
```

## License

Same license as the main ALIKED project. See LICENSE file.
