"""
Export ALIKED PyTorch model to ONNX format

Note: ONNX does NOT support deformable convolutions (DCN). This script creates
a DCN-free version of ALIKED using regular convolutions for ONNX compatibility.
This may result in slightly reduced accuracy compared to the original PyTorch model.
"""

import os
import torch
import torch.nn as nn
import argparse
import logging
from torchvision.models import resnet
from nets.aliked import ALIKED_CFGS
from nets.blocks import ConvBlock, ResBlock, SDDH
from nets.soft_detect import DKD, simple_nms
from nets.padder import InputPadder

# Define native PyTorch implementation of get_patches for ONNX compatibility
def get_patches_lg(
    tensor: torch.Tensor, required_corners: torch.Tensor, ps: int
) -> torch.Tensor:
    """Native PyTorch implementation of patch extraction (ONNX-compatible)"""
    c, h, w = tensor.shape
    corner = (required_corners - ps / 2 + 1).long()
    corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
    corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
    offset = torch.arange(0, ps)

    kw = {"indexing": "ij"} if torch.__version__ >= "1.10" else {}
    x, y = torch.meshgrid(offset, offset, **kw)
    patches = torch.stack((x, y)).permute(2, 1, 0).unsqueeze(2)
    patches = patches.to(corner) + corner[None, None]
    pts = patches.reshape(-1, 2)
    sampled = tensor.permute(1, 2, 0)[tuple(pts.T)[::-1]]
    sampled = sampled.reshape(ps, ps, -1, c)
    assert sampled.shape[:3] == patches.shape[:3]
    return sampled.permute(2, 3, 0, 1)

# Force use of native PyTorch implementation instead of custom CUDA op for ONNX
import nets.blocks as blocks_module
blocks_module.with_compiled_get_patches = False
blocks_module.get_patches_lg = get_patches_lg


class DKD_ONNX(nn.Module):
    """ONNX-compatible keypoint detection module - simplified to avoid data-dependent ops"""
    def __init__(self, radius=2, top_k=-1, scores_th=0.2, n_limit=5000):
        super().__init__()
        self.radius = radius
        self.top_k = top_k
        self.scores_th = scores_th
        self.n_limit = n_limit
        
        # For ONNX, we MUST use top_k mode to avoid data-dependent control flow
        if self.n_limit > 0:
            self.top_k = self.n_limit
        elif self.top_k <= 0:
            self.top_k = 5000  # default
    
    def forward(self, scores_map, sub_pixel=True):
        """
        ONNX-compatible forward pass
        
        Args:
            scores_map: Bx1xHxW score map
            sub_pixel: whether to use sub-pixel refinement (always True for ONNX)
        
        Returns:
            keypoints: BxNx2 in normalized coordinates [-1, 1]
            kptscores: BxN confidence scores
            scoredispersitys: BxN score dispersity (placeholder, returns scores)
        """
        b, c, h, w = scores_map.shape
        
        # NMS
        scores_nograd = scores_map.detach()
        nms_scores = simple_nms(scores_nograd, self.radius * 2 + 1)
        
        # Remove border
        nms_scores[:, :, :self.radius, :] = 0
        nms_scores[:, :, :, :self.radius] = 0
        nms_scores[:, :, h - self.radius:, :] = 0
        nms_scores[:, :, :, w - self.radius:] = 0
        
        # Top-K selection (ONNX-compatible, no data-dependent branches)
        nms_flat = nms_scores.view(b, -1)  # B x (H*W)
        topk_scores, topk_indices = torch.topk(nms_flat, self.top_k, dim=1)  # B x top_k
        
        # Convert flat indices to 2D coordinates (PIXEL coordinates)
        topk_y = (topk_indices // w).float()  # B x top_k
        topk_x = (topk_indices % w).float()   # B x top_k
        
        # Stack as pixel coordinates (will be normalized in post-processing)
        keypoints_pixel = torch.stack([topk_x, topk_y], dim=-1)  # B x top_k x 2
        
        # Sub-pixel refinement (vectorized for ONNX compatibility)
        if sub_pixel:
            # Reshape for batch processing
            kpts_flat = keypoints_pixel.view(b * self.top_k, 2)  # (B*top_k) x 2
            
            # Extract patches - process each batch item
            # Note: Can't fully vectorize due to get_patches_lg API, but minimize loop impact
            patches_list = []
            for i in range(b):
                idx_start = i * self.top_k
                idx_end = (i + 1) * self.top_k
                kpts_batch = kpts_flat[idx_start:idx_end]  # top_k x 2
                score_batch = scores_map[i]  # 1 x H x W
                patches_batch = get_patches_lg(score_batch, kpts_batch, 5)  # top_k x 1 x 5 x 5
                patches_list.append(patches_batch)
            
            patches = torch.cat(patches_list, dim=0)  # (B*top_k) x 1 x 5 x 5
            patches_flat = patches.view(b * self.top_k, 25)  # (B*top_k) x 25
            
            # Find max location in patch
            max_idx = torch.argmax(patches_flat, dim=1)  # (B*top_k)
            offset_y = (max_idx // 5).float() - 2.0  # offset from center
            offset_x = (max_idx % 5).float() - 2.0
            
            # Apply sub-pixel offset in pixel space (damped)
            offset_pixels = torch.stack([offset_x * 0.5, offset_y * 0.5], dim=-1)  # (B*top_k) x 2
            offset_pixels = offset_pixels.view(b, self.top_k, 2)
            keypoints_pixel = keypoints_pixel + offset_pixels
            
            # Clamp to valid range [0, w-1] and [0, h-1]
            keypoints_pixel[:, :, 0] = torch.clamp(keypoints_pixel[:, :, 0], 0, w - 1)
            keypoints_pixel[:, :, 1] = torch.clamp(keypoints_pixel[:, :, 1], 0, h - 1)
        
        # Return topk_scores as keypoint scores
        kptscores = topk_scores  # B x top_k
        scoredispersitys = topk_scores  # placeholder, same as scores
        
        return keypoints_pixel, kptscores, scoredispersitys


class ALIKED_ONNX(nn.Module):
    """ALIKED model with regular convolutions instead of DCN for ONNX export"""
    def __init__(self, model_name, device='cpu', top_k=-1, scores_th=0.2, n_limit=5000):
        super().__init__()
        
        # Get configurations
        c1, c2, c3, c4, dim, K, M = [v for _,v in ALIKED_CFGS[model_name].items()]
        
        # Replace DCN with regular convolutions for ONNX compatibility
        conv_types = ['conv', 'conv', 'conv', 'conv']  # No DCN!
        conv2D = False
        mask = False
        self.device = device
        
        # Build model
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        self.block1 = ConvBlock(3, c1, self.gate, self.norm, conv_type=conv_types[0])
        self.block2 = ResBlock(c1, c2, 1, nn.Conv2d(c1, c2, 1),
                                gate=self.gate,
                                norm_layer=self.norm,
                                conv_type=conv_types[1])
        self.block3 = ResBlock(c2, c3, 1, nn.Conv2d(c2, c3, 1),
                            gate=self.gate,
                            norm_layer=self.norm,
                            conv_type=conv_types[2],
                            mask=mask)
        self.block4 = ResBlock(c3, c4, 1, nn.Conv2d(c3, c4, 1),
                            gate=self.gate,
                            norm_layer=self.norm,
                            conv_type=conv_types[3],
                            mask=mask)
        self.conv1 = resnet.conv1x1(c1, dim // 4)
        self.conv2 = resnet.conv1x1(c2, dim // 4)
        self.conv3 = resnet.conv1x1(c3, dim // 4)
        self.conv4 = resnet.conv1x1(dim, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.score_head = nn.Sequential(resnet.conv1x1(dim, 8), self.gate, resnet.conv3x3(8, 4), self.gate,
                                                resnet.conv3x3(4, 4), self.gate, resnet.conv3x3(4, 1))
        self.desc_head = SDDH(dim, K, M, gate=self.gate, conv2D=conv2D, mask=mask)
        # Use ONNX-compatible DKD module
        self.dkd = DKD_ONNX(radius=2, top_k=top_k, scores_th=scores_th, n_limit=n_limit)
    
    def extract_dense_map(self, image):
        # Pads images such that dimensions are divisible by 32
        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        image = padder.pad(image)

        # Feature encoder
        x1 = self.block1(image)
        x2 = self.pool2(x1)
        x2 = self.block2(x2)
        x3 = self.pool4(x2)
        x3 = self.block3(x3)
        x4 = self.pool4(x3)
        x4 = self.block4(x4)
        
        # Feature aggregation
        x1 = self.gate(self.conv1(x1))
        x2 = self.gate(self.conv2(x2))
        x3 = self.gate(self.conv3(x3))
        x4 = self.gate(self.conv4(x4))
        x2_up = self.upsample2(x2)
        x3_up = self.upsample8(x3)
        x4_up = self.upsample32(x4)
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)
        
        # Score head
        score_map = torch.sigmoid(self.score_head(x1234))
        feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)

        # Unpads images
        feature_map = padder.unpad(feature_map)
        score_map = padder.unpad(score_map)

        return feature_map, score_map

    def forward(self, image):
        b, c, h, w = image.shape
        feature_map, score_map = self.extract_dense_map(image)
        keypoints_pixel, kptscores, scoredispersitys = self.dkd(score_map)  # keypoints in pixel coords
        
        # Convert pixel coordinates to normalized [-1, 1] for SDDH
        # Formula: norm = 2 * pixel / (size - 1) - 1
        h_score, w_score = score_map.shape[2], score_map.shape[3]
        keypoints_norm = torch.stack([
            2.0 * keypoints_pixel[:, :, 0] / (w_score - 1) - 1.0,  # x
            2.0 * keypoints_pixel[:, :, 1] / (h_score - 1) - 1.0   # y
        ], dim=-1)  # B x N x 2
        
        descriptors_list, offsets = self.desc_head(feature_map, keypoints_norm)
        
        # Stack descriptors list into batched tensor
        descriptors = torch.stack(descriptors_list, dim=0)  # B x N x D

        # Return pixel coordinates (easier for inference), descriptors, scores, score_map
        return keypoints_pixel, descriptors, kptscores, score_map


def export_to_onnx(model_name='aliked-n16rot', output_path=None, opset_version=16, simplify=True):
    """
    Export ALIKED model to ONNX format
    
    Args:
        model_name: Model configuration name
        output_path: Path to save ONNX model. If None, saves to models/{model_name}.onnx
        opset_version: ONNX opset version (minimum 16 for grid_sampler support)
        simplify: Whether to simplify the ONNX model using onnxsim
        
    Note:
        DCN (Deformable Convolution) layers are replaced with regular convolutions
        for ONNX compatibility, which may slightly reduce accuracy.
    """
    
    # Set output path
    if output_path is None:
        output_path = f'models/{model_name}.onnx'
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Create ALIKED model WITHOUT DCN for ONNX compatibility
    logging.info(f"Loading PyTorch model: {model_name} (DCN-free version for ONNX)")
    
    model = ALIKED_ONNX(
        model_name=model_name,
        device='cpu',
        top_k=-1,
        scores_th=0.2,
        n_limit=5000
    )
    
    # Load pretrained weights (strict=False to skip DCN layers)
    pretrained_path = f'models/{model_name}.pth'
    if os.path.exists(pretrained_path):
        logging.info(f'Loading pretrained weights from {pretrained_path} (skipping DCN layers)')
        state_dict = torch.load(pretrained_path, 'cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logging.info(f'Skipped {len(missing_keys)} DCN-related keys during weight loading')
    else:
        raise FileNotFoundError(f'Cannot find pretrained model: {pretrained_path}')
    
    model.eval()
    
    # Create dummy input (1, 3, H, W) - use a size divisible by 32
    dummy_h, dummy_w = 480, 640
    dummy_input = torch.randn(1, 3, dummy_h, dummy_w)
    
    logging.info(f"Exporting model with input shape: {dummy_input.shape}")
    logging.info(f"Output path: {output_path}")
    logging.info(f"ONNX opset version: {opset_version}")
    
    # Define input and output names
    input_names = ['image']
    output_names = ['keypoints', 'descriptors', 'scores', 'score_map']
    
    # Dynamic axes for variable input sizes
    dynamic_axes = {
        'image': {0: 'batch', 2: 'height', 3: 'width'},
        'keypoints': {0: 'batch', 1: 'num_keypoints'},
        'descriptors': {0: 'batch', 1: 'num_keypoints'},
        'scores': {0: 'batch', 1: 'num_keypoints'},
        'score_map': {0: 'batch', 2: 'height', 3: 'width'},
    }
    
    # Export to ONNX using legacy JIT-based exporter
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False  # Use legacy JIT-based exporter
        )
    
    logging.info(f"Model exported successfully to {output_path}")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX model verification passed")
        
        # Print model info
        logging.info(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
        logging.info(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")
        
    except ImportError:
        logging.warning("onnx package not found. Install with: pip install onnx")
    except Exception as e:
        logging.error(f"ONNX model verification failed: {e}")
        return False
    
    # Simplify the model
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            logging.info("Simplifying ONNX model...")
            
            onnx_model, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                logging.info(f"Simplified model saved to {output_path}")
            else:
                logging.warning("Simplification check failed, using original model")
        except ImportError:
            logging.warning("onnxsim not found. Install with: pip install onnxsim")
        except Exception as e:
            logging.warning(f"Simplification failed: {e}. Using original model.")
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.info(f"ONNX model size: {file_size_mb:.2f} MB")
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export ALIKED model to ONNX format (DCN-free version)')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'],
                        default='aliked-n16rot', help='Model configuration to export')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for ONNX model (default: models/{model_name}.onnx)')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version (default: 16, minimum 16 for grid_sampler)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX model simplification')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    success = export_to_onnx(
        model_name=args.model,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )
    
    if success:
        logging.info("\n" + "="*60)
        logging.info("Export completed successfully!")
        logging.info("="*60)
        logging.info(f"\nNote: This ONNX model uses regular convolutions instead of DCN.")
        logging.info(f"Accuracy may be slightly lower than the original PyTorch model.")
        logging.info(f"\nTo use the ONNX model:")
        logging.info(f"  python demo_pair_onnx.py <image_dir> --model {args.model}")
    else:
        logging.error("Export failed!")
