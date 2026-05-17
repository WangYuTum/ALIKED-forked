"""
ONNX Runtime wrapper for ALIKED model
"""

import numpy as np
import cv2
import logging


class ALIKED_ONNX:
    """ONNX Runtime inference wrapper for ALIKED"""
    
    def __init__(self, onnx_path, device='cpu', n_limit=5000):
        """
        Initialize ONNX Runtime session
        
        Args:
            onnx_path: Path to ONNX model file
            device: 'cpu' or 'cuda'
            n_limit: Maximum number of keypoints (Note: built into model, this is for reference)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not found. Install with:\n"
                "  pip install onnxruntime  # for CPU\n"
                "  pip install onnxruntime-gpu  # for GPU"
            )
        
        self.n_limit = n_limit
        self.device = device
        
        # Set up providers
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        logging.info(f"Loading ONNX model from: {onnx_path}")
        logging.info(f"Using providers: {providers}")
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get descriptor dimension from output shape
        descriptor_output = self.session.get_outputs()[1]
        # Shape is (1, N, D) or (N, D), get last dimension
        self.descriptor_dim = descriptor_output.shape[-1] if hasattr(descriptor_output.shape[-1], '__int__') else 128
        
        logging.info(f"Model input: {self.input_name}")
        logging.info(f"Model outputs: {self.output_names}")
        logging.info(f"Descriptor dimension: {self.descriptor_dim}")
        logging.info(f"ONNX Runtime initialized successfully")
    
    def preprocess(self, img_rgb):
        """
        Preprocess RGB image for model input
        
        Args:
            img_rgb: RGB image (H, W, 3) as numpy array, uint8
        
        Returns:
            Preprocessed tensor (1, 3, H, W) as float32
        """
        # Convert to float and normalize to [0, 1]
        img = img_rgb.astype(np.float32) / 255.0
        
        # Transpose from HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def run(self, img_rgb):
        """
        Run inference on RGB image
        
        Args:
            img_rgb: RGB image (H, W, 3) as numpy array
        
        Returns:
            Dictionary with:
                - keypoints: (N, 2) array of keypoint coordinates
                - descriptors: (N, D) array of descriptors
                - scores: (N,) array of keypoint scores
                - score_map: (H, W) score map
        """
        # Get original image dimensions
        h, w = img_rgb.shape[:2]
        
        # Preprocess
        img_tensor = self.preprocess(img_rgb)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: img_tensor}
        )
        
        # Parse outputs (ONNX model returns 4 outputs: keypoints, descriptors, scores, score_map)
        # Note: ONNX outputs don't include batch dimension (already squeezed)
        # Expected shapes: (N, 2), (N, D), (N,), (1, 1, H, W)
        keypoints_raw = outputs[0]
        descriptors_raw = outputs[1]
        scores_raw = outputs[2]
        score_map_raw = outputs[3]
        
        # Handle different shape formats
        if len(keypoints_raw.shape) == 3:  # (1, N, 2) - with batch dimension
            keypoints = keypoints_raw[0]  # (N, 2)
            descriptors = descriptors_raw[0]  # (N, D)
            scores = scores_raw[0]  # (N,)
        elif len(keypoints_raw.shape) == 2:  # (N, 2) - batch already removed
            keypoints = keypoints_raw
            descriptors = descriptors_raw
            scores = scores_raw
        else:
            logging.warning(f"Unexpected keypoints shape: {keypoints_raw.shape}, attempting reshape")
            keypoints = keypoints_raw.reshape(-1, 2) if keypoints_raw.size > 0 else np.zeros((0, 2), dtype=np.float32)
            descriptors = descriptors_raw.reshape(-1, descriptors_raw.shape[-1]) if descriptors_raw.size > 0 else np.zeros((0, 128), dtype=np.float32)
            scores = scores_raw.flatten() if scores_raw.size > 0 else np.zeros((0,), dtype=np.float32)
        
        # Score map: (1, 1, H, W) -> (H, W)
        score_map = score_map_raw[0, 0] if len(score_map_raw.shape) == 4 else score_map_raw.squeeze()
        
        # Handle case where no keypoints are detected
        if keypoints.size == 0 or keypoints.shape[0] == 0:
            return {
                'keypoints': np.zeros((0, 2), dtype=np.float32),
                'descriptors': np.zeros((0, self.descriptor_dim), dtype=np.float32),
                'scores': np.zeros((0,), dtype=np.float32),
                'score_map': score_map
            }
        
        # Check if keypoints are already in pixel coordinates or normalized [-1, 1]
        # If max value > 2, they're likely already in pixel coordinates
        if np.abs(keypoints).max() > 2:
            logging.debug("Keypoints appear to be in pixel coordinates already")
            keypoints_denorm = keypoints
        else:
            # Denormalize keypoints from [-1, 1] to pixel coordinates
            keypoints_denorm = np.zeros_like(keypoints)
            keypoints_denorm[:, 0] = (keypoints[:, 0] + 1) * (w - 1) / 2
            keypoints_denorm[:, 1] = (keypoints[:, 1] + 1) * (h - 1) / 2
        
        return {
            'keypoints': keypoints_denorm,
            'descriptors': descriptors,
            'scores': scores,
            'score_map': score_map
        }


def convert_image_to_rgb(image):
    """Convert BGR (OpenCV) image to RGB"""
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
