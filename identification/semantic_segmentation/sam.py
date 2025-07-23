import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from typing import List, Dict
import os
import cv2
import supervision as sv

class SAMSegmentation:
    """
    Class for performing segmentation using the Segment Anything Model (SAM).
    Handles model initialization, image processing, and visualization of results.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize SAM model and automatic mask generator"""
        print(f"\nInitializing SAM with device: {device}")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._initialize_model(device)
        
    def _initialize_model(self, device: str):
        """Initialize the model on the specified device, fallback to CPU if OOM occurs"""
        try:
            self.sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            self.sam.to(device=device)
            self.device = device
        except torch.cuda.OutOfMemoryError:
            print("GPU out of memory! Reverting to CPU mode...")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)
            
        # Initialize the mask generator
        self.mask_generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92
        )
        
    def process_image(self, image_path: str) -> Dict:
        """Process an image and return segmentation masks, resizing if necessary"""
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Resize large images (Prevents high memory usage)
        max_size = 1024
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            result = self.mask_generator.generate(image_rgb)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU out of memory, retrying with CPU...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self._initialize_model("cpu")  # Switch to CPU mode
                result = self.mask_generator.generate(image_rgb)
            else:
                raise e
                
        return result
        
    def visualize_masks(self, image_path: str, result: List[Dict], output_name: str, output_dir: str):
        """Visualize segmentation masks using supervision"""
        image_bgr = cv2.imread(image_path)
        original_size = image_bgr.shape[:2]
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        
        for mask_data in result:
            mask_data['segmentation'] = cv2.resize(
                mask_data['segmentation'].astype(np.uint8),
                (original_size[1], original_size[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
        detections = sv.Detections.from_sam(result)
        annotated_image = mask_annotator.annotate(
            scene=image_bgr,
            detections=detections
        )
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"image_{output_name}.png")
        cv2.imwrite(output_path, annotated_image)
        return output_path
        
    def save_segments_boxes(self, masks: List[Dict], output_path: str):
        """Save binary masks and bounding boxes to a numpy file"""
        binary_masks = []
        boxes = []
        areas = []
        
        for mask_data in masks:
            binary_masks.append(mask_data['segmentation'])
            x, y, w, h = mask_data['bbox']
            boxes.append([x, y, x + w, y + h])
            areas.append(mask_data['area'])
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path,
                 masks=np.array(binary_masks),
                 boxes=np.array(boxes),
                 areas=np.array(areas))