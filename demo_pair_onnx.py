"""
ALIKED image pair matching demo using ONNX Runtime
"""

import os
import cv2
import glob
import logging
import argparse
import numpy as np
from nets.aliked_onnx import ALIKED_ONNX
from copy import deepcopy


class ImageLoader(object):
    def __init__(self, filepath: str):
        self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                      glob.glob(os.path.join(filepath, '*.jpg')) + \
                      glob.glob(os.path.join(filepath, '*.ppm'))
        self.images.sort()
        self.N = len(self.images)
        logging.info(f'Loading {self.N} images')
        self.mode = 'images'

    def __getitem__(self, item):
        filename = self.images[item]
        img = cv2.imread(filename)   
        return img

    def __len__(self):
        return self.N


def mnn_mather(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()


def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv2.circle(out, (x0, y0), radius, color, -1, lineType=cv2.LINE_4)
    return out


def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 matches,
                 radius=2,
                 color=(255, 0, 0),
                 mcolor=(0, 255, 0)):

    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1

        cv2.line(out, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=1,
                     lineType=cv2.LINE_AA)

    return out
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALIKED image pair Demo with ONNX Runtime.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory.')
    parser.add_argument('--model', choices=['aliked-t16', 'aliked-n16', 'aliked-n16rot', 'aliked-n32'], 
                        default="aliked-n16rot",
                        help="The model configuration")
    parser.add_argument('--onnx-path', type=str, default=None,
                        help='Path to ONNX model file. If not provided, uses models/{model}.onnx')
    parser.add_argument('--device', type=str, default='cpu', 
                        help="Running device (default: cpu). Use 'cuda' for GPU with onnxruntime-gpu")
    parser.add_argument('--n_limit', type=int, default=5000,
                        help='Maximum number of keypoints to be detected (default: 5000)')
    parser.add_argument('--resize', type=str, default='640x480', choices=['640x480', '800x600', 'none'],
                        help='Resize images to specified resolution (default: 640x480). Use "none" to keep original size.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse resize option
    if args.resize != 'none':
        target_w, target_h = map(int, args.resize.split('x'))
        logging.info(f"Images will be resized to {target_w}×{target_h}")
    else:
        target_w, target_h = None, None
        logging.info("Images will be kept at original size")
    
    # Set ONNX model path
    if args.onnx_path is None:
        args.onnx_path = f'models/{args.model}.onnx'
    
    # Check if ONNX model exists
    if not os.path.exists(args.onnx_path):
        logging.error(f"ONNX model not found: {args.onnx_path}")
        logging.error(f"Please export the model first:")
        logging.error(f"  python export_onnx.py --model {args.model}")
        exit(1)
    
    # Load ONNX model
    model = ALIKED_ONNX(
        onnx_path=args.onnx_path,
        device=args.device,
        n_limit=args.n_limit
    )
    
    # Load images
    image_loader = ImageLoader(args.input)
    
    logging.info("Press 'space' to start. \nPress 'q' or 'ESC' to stop!")
    
    # Process reference image
    img_ref = image_loader[0]
    
    # Resize if requested
    if target_w is not None:
        img_ref_original = img_ref.copy()
        img_ref = cv2.resize(img_ref, (target_w, target_h))
        logging.info(f"Reference image resized from {img_ref_original.shape[:2][::-1]} to {img_ref.shape[:2][::-1]}")
    
    img_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    pred_ref = model.run(img_rgb)
    kpts_ref = pred_ref['keypoints']
    desc_ref = pred_ref['descriptors']
    
    logging.info(f"Reference image: {len(kpts_ref)} keypoints detected")
    
    # Process remaining images
    for i in range(1, len(image_loader)):
        img = image_loader[i]
        if img is None:
            break
        
        # Resize if requested
        if target_w is not None:
            img = cv2.resize(img, (target_w, target_h))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = model.run(img_rgb)
        kpts = pred['keypoints']
        desc = pred['descriptors']

        logging.info(f"Image {i}: {len(kpts)} keypoints detected")
        
        matches = mnn_mather(desc_ref, desc)
        status = f"matches/keypoints: {len(matches)}/{len(kpts)}"
        
        vis_img = plot_matches(img_ref, img, kpts_ref, kpts, matches)
        
        window_name = f"{args.model} (ONNX)"
        cv2.namedWindow(window_name)
        cv2.setWindowTitle(window_name, f"{args.model} (ONNX): {status}")
        cv2.putText(vis_img, "ONNX Runtime - Press 'q' or 'ESC' to stop.", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, vis_img)
        
        c = cv2.waitKey()
        if c == ord('q') or c == 27:
            break

    logging.info('Finished!')
    logging.info('Press any key to exit!')
    cv2.putText(vis_img, "Finished! Press any key to exit.", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, vis_img)
    cv2.waitKey()
